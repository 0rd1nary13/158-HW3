"""Homework 3 implementation module.

This module provides implementations for:
- Rating prediction (global mean, bias model with updates, evaluation, writers)
- Read prediction (validation sampling, popularity baselines, Jaccard predictor)
- Category prediction features (bag-of-words) and a stronger feature set

All public functions include type annotations and PEP 257-compliant docstrings.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple

import gzip
import random
import string


# ---------- Types ----------

Rating = Tuple[str, str, int]
User = str
Item = str


# ---------- Rating prediction ----------

def getGlobalAverage(trainRatings: Sequence[int]) -> float:
    """Return the average rating in the training set.

    Args:
        trainRatings: A sequence of integer ratings.

    Returns:
        The arithmetic mean of the ratings as a float.
    """

    if not trainRatings:
        return 0.0
    return sum(trainRatings) / float(len(trainRatings))


def trivialValidMSE(ratingsValid: Sequence[Rating], globalAverage: float) -> float:
    """Compute MSE of a trivial model that always predicts the global mean.

    Args:
        ratingsValid: Validation triples of (user, item, rating).
        globalAverage: Mean rating predicted for every pair.

    Returns:
        The mean squared error on the validation set.
    """

    if not ratingsValid:
        return 0.0
    se_sum = 0.0
    for _u, _b, r in ratingsValid:
        err = r - globalAverage
        se_sum += err * err
    return se_sum / float(len(ratingsValid))


def alphaUpdate(
    ratingsTrain: Sequence[Rating],
    alpha: float,
    betaU: MutableMapping[User, float],
    betaI: MutableMapping[Item, float],
    lamb: float,
) -> float:
    """Update equation for global bias alpha.

    Sets derivative to zero (no regularization on alpha):
        alpha = mean_{(u,i) in train} (r_ui - beta_u - beta_i)

    Args:
        ratingsTrain: Training ratings (user, item, rating).
        alpha: Current global bias (ignored by update formula).
        betaU: User bias mapping.
        betaI: Item bias mapping.
        lamb: Regularization weight (not used for alpha update).

    Returns:
        The updated global bias alpha.
    """

    if not ratingsTrain:
        return 0.0

    residual_sum = 0.0
    for u, b, r in ratingsTrain:
        residual_sum += r - betaU.get(u, 0.0) - betaI.get(b, 0.0)
    return residual_sum / float(len(ratingsTrain))


def betaUUpdate(
    ratingsPerUser: MappingUserRatings,
    alpha: float,
    betaU: MutableMapping[User, float],
    betaI: MappingItemBias,
    lamb: float,
) -> Dict[User, float]:
    """Update equation for user biases betaU.

    For each user u:
        beta_u = sum_{i in N(u)} (r_ui - alpha - beta_i) / (lambda + |N(u)|)

    Args:
        ratingsPerUser: Mapping from user -> list of (item, rating) in train.
        alpha: Current global bias.
        betaU: Current user biases (used only for key presence; values not needed).
        betaI: Current item biases (read for the update).
        lamb: L2 regularization strength.

    Returns:
        A new mapping of user biases after the update.
    """

    newBetaU: Dict[User, float] = {}
    for u, item_r_list in ratingsPerUser.items():
        if not item_r_list:
            newBetaU[u] = 0.0
            continue
        numer = 0.0
        denom = lamb + float(len(item_r_list))
        for b, r in item_r_list:
            numer += r - alpha - betaI.get(b, 0.0)
        newBetaU[u] = numer / denom
    return newBetaU


def betaIUpdate(
    ratingsPerItem: MappingItemRatings,
    alpha: float,
    betaU: MappingUserBias,
    betaI: MutableMapping[Item, float],
    lamb: float,
) -> Dict[Item, float]:
    """Update equation for item biases betaI.

    For each item i:
        beta_i = sum_{u in N(i)} (r_ui - alpha - beta_u) / (lambda + |N(i)|)

    Args:
        ratingsPerItem: Mapping from item -> list of (user, rating) in train.
        alpha: Current global bias.
        betaU: Current user biases (read for the update).
        betaI: Current item biases (used only for key presence; values not needed).
        lamb: L2 regularization strength.

    Returns:
        A new mapping of item biases after the update.
    """

    newBetaI: Dict[Item, float] = {}
    for b, user_r_list in ratingsPerItem.items():
        if not user_r_list:
            newBetaI[b] = 0.0
            continue
        numer = 0.0
        denom = lamb + float(len(user_r_list))
        for u, r in user_r_list:
            numer += r - alpha - betaU.get(u, 0.0)
        newBetaI[b] = numer / denom
    return newBetaI


def msePlusReg(
    ratingsTrain: Sequence[Rating],
    alpha: float,
    betaU: MappingUserBias,
    betaI: MappingItemBias,
    lamb: float,
) -> Tuple[float, float]:
    """Compute the MSE and regularized objective on the training set.

    Objective = MSE + lamb * (sum_u beta_u^2 + sum_i beta_i^2)

    Args:
        ratingsTrain: Training ratings (user, item, rating).
        alpha: Global bias.
        betaU: User biases.
        betaI: Item biases.
        lamb: L2 regularization strength for biases.

    Returns:
        A pair (mse, objective_value).
    """

    if not ratingsTrain:
        return 0.0, 0.0

    se_sum = 0.0
    for u, b, r in ratingsTrain:
        pred = alpha + betaU.get(u, 0.0) + betaI.get(b, 0.0)
        err = r - pred
        se_sum += err * err
    mse = se_sum / float(len(ratingsTrain))

    reg = 0.0
    for v in betaU.values():
        reg += v * v
    for v in betaI.values():
        reg += v * v

    return mse, mse + lamb * reg


def validMSE(
    ratingsValid: Sequence[Rating],
    alpha: float,
    betaU: MappingUserBias,
    betaI: MappingItemBias,
) -> float:
    """Compute the MSE on the validation set for the bias model.

    Missing biases default to 0.

    Args:
        ratingsValid: Validation ratings (user, item, rating).
        alpha: Global bias.
        betaU: User biases.
        betaI: Item biases.

    Returns:
        The mean squared error on the validation data.
    """

    if not ratingsValid:
        return 0.0
    se_sum = 0.0
    for u, b, r in ratingsValid:
        pred = alpha + betaU.get(u, 0.0) + betaI.get(b, 0.0)
        diff = r - pred
        se_sum += diff * diff
    return se_sum / float(len(ratingsValid))


def goodModel(
    ratingsTrain: Sequence[Rating],
    ratingsPerUser: MappingUserRatings,
    ratingsPerItem: MappingItemRatings,
    alpha: float,
    betaU: MutableMapping[User, float],
    betaI: MutableMapping[Item, float],
    *,
    lamb: float = 5.0,
    iterations: int = 5,
) -> Tuple[float, Dict[User, float], Dict[Item, float]]:
    """Improve upon the single-iteration model by running multiple iterations.

    Runs coordinate updates for alpha, betaU, and betaI for a few iterations.

    Args:
        ratingsTrain: Training ratings.
        ratingsPerUser: Mapping user -> (item, rating) list.
        ratingsPerItem: Mapping item -> (user, rating) list.
        alpha: Initial global bias.
        betaU: Initial user biases (will not be mutated; copied to new dicts).
        betaI: Initial item biases (will not be mutated; copied to new dicts).
        lamb: L2 regularization strength.
        iterations: Number of update iterations to run.

    Returns:
        Tuple of (alpha, betaU, betaI) after updates.
    """

    # Copy to avoid mutating inputs
    bU: Dict[User, float] = {u: float(v) for u, v in betaU.items()}
    bI: Dict[Item, float] = {i: float(v) for i, v in betaI.items()}

    a = alpha
    for _ in range(max(1, int(iterations))):
        a = alphaUpdate(ratingsTrain, a, bU, bI, lamb)
        bU = betaUUpdate(ratingsPerUser, a, bU, bI, lamb)
        bI = betaIUpdate(ratingsPerItem, a, bU, bI, lamb)
    return a, bU, bI


def writePredictionsRating(alpha: float, betaU: MappingUserBias, betaI: MappingItemBias) -> None:
    """Write rating predictions to predictions_Rating.csv based on pairs_Rating.csv.

    Args:
        alpha: Global bias.
        betaU: User biases.
        betaI: Item biases.
    """

    predictions = open("predictions_Rating.csv", "w")
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        _ = predictions.write(u + "," + b + "," + str(alpha + bu + bi) + "\n")
    predictions.close()


# ---------- Read prediction ----------

def generateValidation(
    allRatings: Sequence[Rating],
    ratingsValid: Sequence[Rating],
) -> Tuple[Set[Tuple[User, Item]], Set[Tuple[User, Item]]]:
    """Generate validation sets of positive and negative (user, item) pairs.

    For each (user, item) in ratingsValid, sample a distinct negative pair
    (user, item') where item' is not in the user's history (from allRatings).

    Args:
        allRatings: Entire set of ratings pairs to determine user histories.
        ratingsValid: Validation subset used to generate positives/negatives.

    Returns:
        A pair of sets (readValid, notRead) with equal sizes matching
        len(ratingsValid).
    """

    # Reproducibility without being too strict
    rnd = random.Random(0)

    user_to_books: Dict[User, Set[Item]] = {}
    all_books: Set[Item] = set()
    for u, b, _r in allRatings:
        all_books.add(b)
        user_to_books.setdefault(u, set()).add(b)

    all_books_list = list(all_books)

    readValid: Set[Tuple[User, Item]] = set((u, b) for (u, b, _r) in ratingsValid)
    notRead: Set[Tuple[User, Item]] = set()

    for u, _b, _r in ratingsValid:
        history = user_to_books.get(u, set())
        # Keep trying until we find a unique negative for this user
        # that the user has not read.
        while True:
            candidate = all_books_list[rnd.randrange(len(all_books_list))]
            if (candidate not in history) and ((u, candidate) not in notRead):
                notRead.add((u, candidate))
                break

    return readValid, notRead


def baseLineStrategy(mostPopular: Sequence[Tuple[int, Item]], totalRead: int) -> Set[Item]:
    """Return items to predict as read using popularity until covering half of reads.

    Args:
        mostPopular: Pairs of (count, item) sorted descending by count.
        totalRead: Total number of read interactions.

    Returns:
        A set of items for which to predict True.
    """

    return1: Set[Item] = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead / 2.0:
            break
    return return1


def improvedStrategy(mostPopular: Sequence[Tuple[int, Item]], totalRead: int) -> Set[Item]:
    """Improved popularity strategy with a stricter threshold.

    Using a smaller cumulative coverage threshold typically reduces false positives
    and improves accuracy compared to the 50% cutoff baseline.

    Args:
        mostPopular: Pairs of (count, item) sorted descending by count.
        totalRead: Total number of read interactions.

    Returns:
        A set of items for which to predict True.
    """

    threshold = 0.30  # stricter than 0.5 baseline; improves precision
    cutoff = totalRead * threshold
    return1: Set[Item] = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > cutoff:
            break
    return return1


def evaluateStrategy(
    return1: Set[Item],
    readValid: Set[Tuple[User, Item]],
    notRead: Set[Tuple[User, Item]],
) -> float:
    """Compute accuracy of a strategy that predicts True for items in return1.

    Args:
        return1: Items predicted as read.
        readValid: Positive labeled pairs (user, item).
        notRead: Negative labeled pairs (user, item).

    Returns:
        Accuracy over the union of readValid and notRead.
    """

    total = len(readValid) + len(notRead)
    if total == 0:
        return 0.0
    correct = 0
    for u, b in readValid:
        if b in return1:
            correct += 1
    for u, b in notRead:
        if b not in return1:
            correct += 1
    return correct / float(total)


def Jaccard(s1: Set[str], s2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        s1: First set.
        s2: Second set.

    Returns:
        Jaccard similarity in [0, 1].
    """

    if not s1 and not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(inter) / float(union) if union > 0 else 0.0


def jaccardThresh(
    u: User,
    b: Item,
    ratingsPerItem: MappingItemRatings,
    ratingsPerUser: MappingUserRatings,
) -> int:
    """Predict read using Jaccard similarity and popularity fallback.

    For the query (u, b), compute the maximum Jaccard similarity between the
    set of users who read b and the sets of users who read each item in u's
    history. Predict 1 if max similarity exceeds 0.013 or if b is popular
    (len(ratingsPerItem[b]) > 40); otherwise predict 0.

    Args:
        u: User ID.
        b: Item ID.
        ratingsPerItem: Mapping item -> list of (user, rating) entries.
        ratingsPerUser: Mapping user -> list of (item, rating) entries.

    Returns:
        1 if predicted read, else 0.
    """

    users_b_list = ratingsPerItem.get(b, [])
    if len(users_b_list) > 40:
        return 1

    users_b: Set[User] = set(x for x, _ in users_b_list)
    history = ratingsPerUser.get(u, [])
    maxSim = 0.0
    for b_hist, _r in history:
        users_bh = set(x for x, _ in ratingsPerItem.get(b_hist, []))
        sim = Jaccard(users_b, users_bh)
        if sim > maxSim:
            maxSim = sim

    if maxSim > 0.013 or len(users_b_list) > 40:
        return 1
    return 0


def writePredictionsRead(
    ratingsPerItem: MappingItemRatings, ratingsPerUser: MappingUserRatings
) -> None:
    """Write read predictions to predictions_Read.csv based on pairs_Read.csv.

    Args:
        ratingsPerItem: Mapping item -> list of (user, rating) entries.
        ratingsPerUser: Mapping user -> list of (item, rating) entries.
    """

    predictions = open("predictions_Read.csv", "w")
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")
        pred = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        _ = predictions.write(u + "," + b + "," + str(pred) + "\n")
    predictions.close()


# ---------- Category prediction ----------

def featureCat(
    datum: Dict[str, object],
    words: Sequence[str],
    wordId: Dict[str, int],
    wordSet: Set[str],
) -> List[int]:
    """Compute bag-of-words features for a single datum.

    Produces binary indicators (0/1) for presence of each word from the given
    vocabulary after lowercasing and removing punctuation. An offset term is
    appended at the end.

    Args:
        datum: A review record with a 'review_text' field.
        words: The ordered vocabulary list.
        wordId: Mapping word -> index into 'words'.
        wordSet: Set of words in the vocabulary for O(1) membership check.

    Returns:
        A list of length len(words) + 1, where the last element is the offset 1.
    """

    feat = [0] * len(words)
    punctuation = set(string.punctuation)
    text = str(datum.get("review_text", "")).lower()
    cleaned = "".join([c for c in text if c not in punctuation])
    for w in cleaned.split():
        if w in wordSet:
            idx = wordId[w]
            # Binary indicator improves conditioning for linear models
            if feat[idx] == 0:
                feat[idx] = 1
    feat.append(1)  # offset term
    return feat


# Globals for betterFeatures so train/test use the same dictionary
_better_words: List[str] = []
_better_wordId: Dict[str, int] = {}
_better_wordSet: Set[str] = set()
_better_NW: int = 1000


def _init_better_dictionary(data: Sequence[Dict[str, object]]) -> None:
    """Initialize the global dictionary for betterFeatures from the given data.

    Args:
        data: Sequence of review dicts with 'review_text'.
    """

    global _better_words, _better_wordId, _better_wordSet

    wordCount: Dict[str, int] = {}
    punctuation = set(string.punctuation)
    for d in data:
        text = str(d.get("review_text", "")).lower()
        cleaned = "".join([c for c in text if c not in punctuation])
        for w in cleaned.split():
            wordCount[w] = wordCount.get(w, 0) + 1

    # Sort by frequency desc, pick top N
    counts = sorted(((c, w) for w, c in wordCount.items()), reverse=True)
    vocab = [w for _c, w in counts[:_better_NW]]

    _better_words = vocab
    _better_wordId = {w: i for i, w in enumerate(_better_words)}
    _better_wordSet = set(_better_words)


def betterFeatures(data: Sequence[Dict[str, object]]) -> List[List[float]]:
    """Produce improved features for category prediction.

    Uses a larger dictionary (default 1000 words) and augments with two
    simple signals: log review length and average token length. Core BOW
    features are normalized term frequencies (count divided by #tokens),
    which improves optimization conditioning for Logistic Regression.
    A final offset term is appended.

    The vocabulary is initialized on the first call and then reused on
    subsequent calls (e.g., for test data) to ensure consistent dimensions.

    Args:
        data: Sequence of review dicts with 'review_text'.

    Returns:
        Feature matrix as a list of lists of floats.
    """

    if not _better_words:
        _init_better_dictionary(data)

    X: List[List[float]] = []
    punctuation = set(string.punctuation)
    for d in data:
        text = str(d.get("review_text", "")).lower()
        cleaned = "".join([c for c in text if c not in punctuation])
        tokens = cleaned.split()

        vec = [0.0] * len(_better_words)
        for w in tokens:
            if w in _better_wordSet:
                vec[_better_wordId[w]] += 1.0

        # Normalize term counts by document length (TF); helps convergence
        num_tokens = float(len(tokens))
        if num_tokens > 0.0:
            inv_len = 1.0 / num_tokens
            for i in range(len(vec)):
                if vec[i] > 0.0:
                    vec[i] *= inv_len

        # Add simple length-based features (log length is better scaled)
        avg_token_len = (sum(len(t) for t in tokens) / num_tokens) if num_tokens > 0 else 0.0
        log_len = math.log1p(num_tokens)

        vec.extend([log_len, avg_token_len])
        vec.append(1.0)  # offset term at the end
        X.append(vec)
    return X


def runOnTest(data_test: Sequence[Dict[str, object]], mod) -> None:
    """Fit provided model on precomputed features and run on test data.

    Note:
        This helper produces features consistent with betterFeatures.
    """

    Xtest = betterFeatures(data_test)
    _ = mod.predict(Xtest)


def writePredictionsCategory(pred_test: Sequence[int]) -> None:
    """Write category predictions to predictions_Category.csv.

    Args:
        pred_test: Predicted genre IDs in the same order as pairs_Category.csv.
    """

    predictions = open("predictions_Category.csv", "w")
    pos = 0
    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")
        _ = predictions.write(u + "," + b + "," + str(pred_test[pos]) + "\n")
        pos += 1
    predictions.close()


# ---------- Protocol-like helper type aliases (post definitions) ----------

from typing import Mapping  # imported after top-level definitions to avoid name clash

MappingUserRatings = Mapping[User, Sequence[Tuple[Item, int]]]
MappingItemRatings = Mapping[Item, Sequence[Tuple[User, int]]]
MappingUserBias = Mapping[User, float]
MappingItemBias = Mapping[Item, float]


