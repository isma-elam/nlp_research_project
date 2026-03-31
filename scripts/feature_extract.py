"""Extraction de features explicables pour le japonais.

Entrées :
- data/json/jlpt_vocab.json
- data/json/joyo.json
- data/json/jlpt_grammar.json

Sortie :
- dictionnaire de features numériques pour un texte.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    # Dépendance optionnelle (recommandée) : tokenisation + lemmatisation.
    # Installation : pip install sudachipy sudachidict_core
    from sudachipy import dictionary as _sudachi_dictionary  # type: ignore
    from sudachipy import tokenizer as _sudachi_tokenizer  # type: ignore

    _HAS_SUDACHI = True
except Exception:  # pragma: no cover
    _sudachi_dictionary = None
    _sudachi_tokenizer = None
    _HAS_SUDACHI = False

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "data" / "json"
VOCAB_JSON = JSON_DIR / "jlpt_vocab.json"
KANJI_JSON = JSON_DIR / "joyo.json"
KANJI_META_JSON = JSON_DIR / "joyo_meta.json"
GRAMMAR_JSON = JSON_DIR / "jlpt_grammar.json"

LEVELS = ["N5", "N4", "N3", "N2", "N1"]
LEVEL_RANK = {"N1": 1, "N2": 2, "N3": 3, "N4": 4, "N5": 5}

# Règles de filtrage pour réduire les faux-positifs par sous-chaîne.
# Le japonais n'a pas d'espaces : un simple `in` peut sur-matcher
# (surtout pour des entrées d'1 caractère comme `角` ou des particules comme `に`).
MIN_VOCAB_LEN = 2
MIN_GRAMMAR_LEN = 2

# Stoplist minimale pour des mots ultra-fréquents qui ne doivent pas piloter
# la difficulté JLPT quand ils matchent comme "vocab".
VOCAB_STOPLIST = {
    "する",  # trop fréquent ; match aussi dans beaucoup d'expressions
}

# Les particules sont sur 1 caractère et seraient normalement filtrées par MIN_GRAMMAR_LEN.
# On les traite explicitement comme signaux de grammaire N5.
GRAMMAR_PARTICLES_N5 = {
    "を",
    "は",
    "が",
    "に",
    "で",
    "へ",
    "と",
    "も",
    "や",
    "の",
}


_GODAN_MASU_ENDING = {
    # masu-stem last kana -> dictionary ending
    "い": "う",
    "き": "く",
    "ぎ": "ぐ",
    "し": "す",
    "ち": "つ",
    "に": "ぬ",
    "び": "ぶ",
    "み": "む",
    "り": "る",
}


PARTICLE_WHITELIST = {
    "を",
    "は",
    "が",
    "に",
    "で",
    "と",
    "も",
    "へ",
    "の",
}


_GODAN_I_TO_U = {
    "い": "う",
    "き": "く",
    "ぎ": "ぐ",
    "し": "す",
    "じ": "ず",
    "ち": "つ",
    "ぢ": "づ",
    "に": "ぬ",
    "ひ": "ふ",
    "び": "ぶ",
    "ぴ": "ぷ",
    "み": "む",
    "り": "る",
}


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text))


def _expand_grammar_variants(pattern: str) -> List[str]:
    """Déplie des patterns de grammaire qui encodent des alternatives.

    De nombreuses entrées JLPT utilisent des séparateurs comme '・' ou '/' pour
    représenter des formes équivalentes (ex: 'だ・です', 'すら / ですら'). Si on les
    garde telles quelles, un matching par sous-chaîne ne matchera jamais le texte.

    Ici on ne fait qu'expanser ; le filtrage (longueur min, vides) est géré par les appelants.
    """

    p = _normalize_text(pattern).strip()
    if not p:
        return []

    # Découper sur des séparateurs d'alternatives courants.
    parts = re.split(r"\s*[/／・]\s*", p)
    # Enlever les espaces et wrappers triviaux.
    out: List[str] = []
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        out.append(part)
    return out


@lru_cache(maxsize=32)
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON manquant : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_punct(ch: str) -> bool:
    """Vrai si `ch` est une ponctuation Unicode."""

    # Les catégories Unicode commençant par 'P' sont de la ponctuation.
    return unicodedata.category(ch).startswith("P")


def _is_kanji(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _standalone_kanji_set(text: str) -> Set[str]:
    """Kanji that are not adjacent to another kanji.

    Matches the intended rule:
    - if kanji are side-by-side, treat them as part of a vocab compound,
      not as isolated kanji difficulty signals.
    """

    s = str(text)
    out: Set[str] = set()
    for i, ch in enumerate(s):
        if not _is_kanji(ch):
            continue
        left_is_kanji = i > 0 and _is_kanji(s[i - 1])
        right_is_kanji = i + 1 < len(s) and _is_kanji(s[i + 1])
        if left_is_kanji or right_is_kanji:
            continue
        out.add(ch)
    return out


def _effective_len_unique(text: str) -> int:
    """Proxy de longueur qui ignore les répétitions.

    Compte les caractères uniques hors espaces/ponctuation. Ça évite que la répétition
    d'un même mot/kanji gonfle artificiellement la "difficulté".
    """

    chars = {ch for ch in str(text) if (not ch.isspace()) and (not _is_punct(ch))}
    return int(len(chars))


def _char_stats(text: str) -> Dict[str, float]:
    # IMPORTANT : les ratios sont calculés hors ponctuation et espaces.
    counted_chars = [ch for ch in text if (not ch.isspace()) and (not _is_punct(ch))]
    total = max(len(counted_chars), 1)

    kanji = sum(1 for ch in counted_chars if "\u4e00" <= ch <= "\u9fff")
    hira = sum(1 for ch in counted_chars if "\u3040" <= ch <= "\u309f")
    kata = sum(1 for ch in counted_chars if "\u30a0" <= ch <= "\u30ff")
    latin = sum(1 for ch in counted_chars if "a" <= ch.lower() <= "z")
    digits = sum(1 for ch in counted_chars if ch.isdigit())

    # Règle: pour vocab/kanji répétés, on ne recompte pas.
    # On applique la même idée aux signaux de "volume" et de ponctuation.
    unique_punct = {ch for ch in text if _is_punct(ch)}
    punct_count = float(len(unique_punct))
    comma_count = float(1 if ("、" in text or "," in text) else 0)

    # Longueur effective: unique chars (hors espaces/ponctuation), puis plafonnée.
    len_cap = float(min(_effective_len_unique(text), 120))

    denom = max(_effective_len_unique(text), 1)

    return {
        "len_chars": len_cap,
        "ratio_kanji": kanji / total,
        "ratio_hira": hira / total,
        "ratio_kata": kata / total,
        "ratio_latin": latin / total,
        "ratio_digits": digits / total,
        "punct_count": punct_count,
        "comma_count": comma_count,
        "ratio_punct": punct_count / denom,
    }


def _sentence_stats(text: str) -> Dict[str, float]:
    # Séparation simple par ponctuation japonaise/latine
    sentences = [s for s in re.split(r"[。！？!?]", text) if s.strip()]
    count = len(sentences)
    # Longueur "effective" par phrase (unique chars), pour ignorer les répétitions.
    eff_lens = [_effective_len_unique(s) for s in sentences]
    avg_len = float(sum(eff_lens) / max(count, 1))
    avg_len_cap = float(min(avg_len, 80.0))
    return {
        "sent_count": float(count),
        "sent_avg_len": avg_len_cap,
    }


def _prepare_vocab(vocab_json: dict) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for level in LEVELS:
        for w in vocab_json.get(level, []):
            if not w:
                continue
            w_norm = _normalize_text(str(w)).strip()
            if not w_norm:
                continue
            if len(w_norm) < MIN_VOCAB_LEN:
                # Autoriser un vocab 1-kanji (ex: 猫) tout en rejetant les kana/particules
                # d'1 caractère qui causent beaucoup de faux-positifs.
                if not (len(w_norm) == 1 and "\u4e00" <= w_norm <= "\u9fff"):
                    continue
            if w_norm in VOCAB_STOPLIST:
                continue
            pairs.append((w_norm, level))
    return pairs


@lru_cache(maxsize=1)
def _get_vocab_sets_by_level() -> Dict[str, Set[str]]:
    """Structure de membership rapide pour les lookups vocab."""

    vocab_json = _load_json(VOCAB_JSON)
    out: Dict[str, Set[str]] = {}
    for lvl in LEVELS:
        out[lvl] = {str(w) for w in vocab_json.get(lvl, []) if w}
    return out


@lru_cache(maxsize=1)
def _get_vocab_set_all() -> Set[str]:
    sets = _get_vocab_sets_by_level()
    all_words: Set[str] = set()
    for s in sets.values():
        all_words |= s
    # Normalize once (same normalizer used elsewhere)
    return {
        _normalize_text(w).strip()
        for w in all_words
        if _normalize_text(w).strip() and (_normalize_text(w).strip() not in VOCAB_STOPLIST)
    }


@lru_cache(maxsize=1)
def _get_sudachi_tokenizer():
    if not _HAS_SUDACHI:
        return None
    return _sudachi_dictionary.Dictionary().create()


def _sudachi_lemmas(text: str) -> List[str]:
    """Renvoie des formes dictionnaire (lemmes) via SudachiPy si disponible.

    Si SudachiPy n'est pas installé, renvoie une liste vide.
    """

    tok = _get_sudachi_tokenizer()
    if tok is None:
        return []

    out: List[str] = []
    try:
        mode = _sudachi_tokenizer.Tokenizer.SplitMode.C  # type: ignore[attr-defined]
        for m in tok.tokenize(str(text), mode):
            lemma = str(m.dictionary_form() or m.normalized_form() or m.surface())
            lemma = _normalize_text(lemma).strip()
            if lemma:
                out.append(lemma)
    except Exception:
        return []

    return out


def _infer_vocab_from_te_ta(text: str) -> Set[str]:
    """Heuristique légère : forme en 〜て / 〜た (ichidan) -> 〜る.

    Exemple : 食べて -> 食べる, 見た -> 見る
    (Les godan sont trop ambigus sans analyse morphologique.)
    """

    vocab_all = _get_vocab_set_all()
    candidates: Set[str] = set()

    segments = re.split(r"[。！？!?、,\s]+|[をはがにでへともや]", str(text))
    segments = [s for s in segments if s]
    for seg in segments:
        m = re.search(r"(.+?)[てた]$", seg)
        if not m:
            continue
        stem = _normalize_text(m.group(1))
        if not stem:
            continue
        cand = _normalize_text(stem + "る")
        if cand in vocab_all:
            candidates.add(cand)

    return candidates


def _infer_vocab_candidates(text: str) -> Set[str]:
    """Candidats vocabulaire (forme dictionnaire) déduits du texte.

    On combine :
    - lemmes SudachiPy (si dispo)
    - heuristique 〜ます (toujours dispo)
    - heuristique minimale 〜て/〜た (ichidan)
    """

    out: Set[str] = set()

    for lemma in _sudachi_lemmas(text):
        if lemma and (lemma not in VOCAB_STOPLIST):
            out.add(lemma)

    for tok in _infer_vocab_from_masu_forms(text):
        if tok and (tok not in VOCAB_STOPLIST):
            out.add(_normalize_text(tok).strip())

    out |= _infer_vocab_from_te_ta(text)
    out.discard("")
    return out


@lru_cache(maxsize=1)
def _get_vocab_token_to_level() -> Dict[str, str]:
    vocab_json = _load_json(VOCAB_JSON)
    token_to_level: Dict[str, str] = {}
    for lvl in LEVELS:
        for w in vocab_json.get(lvl, []):
            if not w:
                continue
            tok = _normalize_text(str(w)).strip()
            if not tok:
                continue
            if tok in VOCAB_STOPLIST:
                continue
            prev = token_to_level.get(tok)
            if prev is None or LEVEL_RANK[lvl] < LEVEL_RANK[prev]:
                token_to_level[tok] = lvl
    return token_to_level


def _prepare_grammar(grammar_json: dict) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for lvl in LEVELS:
        for p in grammar_json.get(lvl, []):
            if not p:
                continue
            for p_norm in _expand_grammar_variants(str(p)):
                if not p_norm:
                    continue
                if len(p_norm) < MIN_GRAMMAR_LEN:
                    continue
                pairs.append((p_norm, lvl))
    # Dé-doublonnage (pattern, niveau) pour garder des comptes stables.
    pairs = list(dict.fromkeys(pairs))

    # Ajout des particules (1 caractère) en patterns N5.
    for p in sorted(GRAMMAR_PARTICLES_N5):
        pairs.append((p, "N5"))
    return list(dict.fromkeys(pairs))


@lru_cache(maxsize=1)
def _get_vocab_pairs() -> List[Tuple[str, str]]:
    vocab_json = _load_json(VOCAB_JSON)
    return _prepare_vocab(vocab_json)


def _infer_vocab_from_masu_forms(text: str) -> List[str]:
    """Infère des formes dictionnaire (lemme) à partir des conjugaisons polies en 〜ます.

    Exemples :
    - 飲みます -> 飲む (godan)
    - 行きます -> 行く (godan)
    - 食べます -> 食べる (ichidan)
    - 勉強します -> 勉強する (suru)

    Intentionnellement minimal (sans tokeniseur complet).
    """

    inferred: Set[str] = set()

    # Match sequences ending in masu-like suffixes.
    for stem, _suffix in re.findall(r"([\u3040-\u30ff\u4e00-\u9fff]{1,12})(ませんでした|ません|ました|ましょう|ます)", text):
        stem = str(stem)
        if not stem:
            continue

        # suru verbs: ...し + ます
        if stem.endswith("し"):
            inferred.add(stem[:-1] + "する")
            continue

        last = stem[-1]

        # Si la dernière char est un kanji (ex: 見ます), l'hypothèse ichidan (〜る) est fréquente.
        if "\u4e00" <= last <= "\u9fff":
            inferred.add(stem + "る")
            continue

        # Hypothèse godan via la terminaison en "ligne i"
        mapped = _GODAN_I_TO_U.get(last)
        if mapped:
            inferred.add(stem[:-1] + mapped)

        # Hypothèse ichidan
        inferred.add(stem + "る")

    return sorted(inferred, key=len, reverse=True)


def _match_vocab_with_inference(text: str, vocab_pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    matched: Dict[str, str] = {}

    # Direct substring matches.
    for token, level in vocab_pairs:
        if token in text:
            prev = matched.get(token)
            if prev is None or LEVEL_RANK[level] < LEVEL_RANK[prev]:
                matched[token] = level

    def _is_vocab_candidate_ok(tok: str) -> bool:
        tok = _normalize_text(str(tok)).strip()
        if not tok:
            return False
        if tok in VOCAB_STOPLIST:
            return False
        # Évite que des particules (1 char) soient traitées comme "vocab".
        if tok in PARTICLE_WHITELIST:
            return False
        # Applique le même garde-fou que _prepare_vocab : vocab très court = faux-positifs.
        if len(tok) < MIN_VOCAB_LEN:
            return bool(len(tok) == 1 and _is_kanji(tok))
        return True

    # Candidats (formes dictionnaire) issus de SudachiPy + heuristiques.
    token_to_level = _get_vocab_token_to_level()
    for tok in _infer_vocab_candidates(text):
        if not _is_vocab_candidate_ok(tok):
            continue
        lvl = token_to_level.get(tok)
        if not lvl:
            continue
        prev = matched.get(tok)
        if prev is None or LEVEL_RANK[lvl] < LEVEL_RANK[prev]:
            matched[tok] = lvl

    kept = _keep_longest_non_substrings(list(matched.keys()))
    counts: Dict[str, int] = dict.fromkeys(LEVELS, 0)  # type: ignore[assignment]
    by_level: Dict[str, List[str]] = {lvl: [] for lvl in LEVELS}
    for token in kept:
        lvl = matched[token]
        counts[lvl] += 1
        by_level[lvl].append(token)
    for lvl in LEVELS:
        by_level[lvl] = sorted(by_level[lvl], key=len, reverse=True)
    return counts, by_level


@lru_cache(maxsize=1)
def _get_grammar_pairs() -> List[Tuple[str, str]]:
    grammar_json = _load_json(GRAMMAR_JSON)
    return _prepare_grammar(grammar_json)


def _keep_longest_non_substrings(strings: List[str]) -> List[str]:
    strings_sorted = sorted(strings, key=len, reverse=True)
    kept: List[str] = []
    for s in strings_sorted:
        if any(s in longer for longer in kept):
            continue
        kept.append(s)
    return kept


def _match_presence_pruned(text: str, pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    matched: Dict[str, str] = {}
    for token, level in pairs:
        if token in text:
            prev = matched.get(token)
            if prev is None or LEVEL_RANK[level] < LEVEL_RANK[prev]:
                matched[token] = level

    kept = _keep_longest_non_substrings(list(matched.keys()))
    counts: Dict[str, int] = dict.fromkeys(LEVELS, 0)  # type: ignore[assignment]
    by_level: Dict[str, List[str]] = {lvl: [] for lvl in LEVELS}
    for token in kept:
        lvl = matched[token]
        counts[lvl] += 1
        by_level[lvl].append(token)
    # Sortie déterministe (utile pour les diffs)
    for lvl in LEVELS:
        by_level[lvl] = sorted(by_level[lvl], key=len, reverse=True)
    return counts, by_level


def _count_presence_pruned(text: str, pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    counts, _ = _match_presence_pruned(text, pairs)
    return counts


def _count_presence_pruned_with_tokens(
    text: str, pairs: List[Tuple[str, str]]
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    # Pour le vocab on peut enrichir les matchs (ex: inférence de la forme dictionnaire depuis 〜ます).
    if pairs is _get_vocab_pairs():
        counts, by_level = _match_vocab_with_inference(text, pairs)
    else:
        counts, by_level = _match_presence_pruned(text, pairs)
    kept: List[Tuple[str, str]] = []
    for lvl in LEVELS:
        for tok in by_level.get(lvl, []):
            kept.append((tok, lvl))
    # Ordre déterministe (plus long d'abord), dé-doublonnage par (token, niveau)
    kept = sorted(set(kept), key=lambda x: len(x[0]), reverse=True)
    return counts, kept


def _level_ratios(prefix: str, counts: Dict[str, int], total: int) -> Dict[str, float]:
    denom = float(max(total, 1))
    return {f"{prefix}_ratio_{lvl}": float(counts[lvl]) / denom for lvl in LEVELS}


def _max_level_num(counts: Dict[str, int]) -> float:
    # Renvoie le niveau le plus difficile présent sous forme numérique (1=difficile, 5=facile), sinon 0.
    for lvl in ["N1", "N2", "N3", "N4", "N5"]:
        if counts.get(lvl, 0) > 0:
            return float(int(lvl[1:]))
    return 0.0


def _vocab_features(text: str, vocab_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    counts, _by_level = _match_vocab_with_inference(text, vocab_pairs)
    matched_total = int(sum(counts.values()))

    features = {f"vocab_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["vocab_total"] = float(matched_total)
    features.update(_level_ratios("vocab", counts, matched_total))
    features["vocab_max_level_num"] = _max_level_num(counts)
    return features


def _extract_vocab_matched_tokens(text: str, vocab_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    _, tokens = _count_presence_pruned_with_tokens(text, vocab_pairs)
    return tokens


def _particle_features(text: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for p in sorted(PARTICLE_WHITELIST):
        feats[f"particle_{p}"] = float(1.0 if p in text else 0.0)
    return feats


def _kanji_set_from_tokens(tokens: List[Tuple[str, str]], *, allow_levels: Set[str]) -> Set[str]:
    covered: Set[str] = set()
    for tok, lvl in tokens:
        if lvl not in allow_levels:
            continue
        for ch in tok:
            if _is_kanji(ch):
                covered.add(ch)
    return covered


def _kanji_features(text: str, kanji_map: Dict[str, int], *, exclude_kanji: Set[str] | None = None) -> Dict[str, float]:
    # IMPORTANT : compte par TYPE de kanji (unique), pas par occurrence.
    # Répéter le même kanji 20 fois ne doit pas "augmenter la difficulté".
    counts = dict.fromkeys(LEVELS, 0)
    exclude = exclude_kanji or set()
    base = _standalone_kanji_set(text)
    unique_kanji: Set[str] = {ch for ch in base if (ch in kanji_map) and (ch not in exclude)}
    for ch in unique_kanji:
        lvl_num = kanji_map[ch]
        lvl = f"N{lvl_num}"
        if lvl in counts:
            counts[lvl] += 1

    features = {f"kanji_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["kanji_total"] = float(len(unique_kanji))
    features.update(_level_ratios("kanji", counts, len(unique_kanji)))
    features["kanji_max_level_num"] = _max_level_num(counts)
    return features


def _collect_kanji_meta_lists(
    text: str, kanji_meta: Dict[str, Dict[str, int]], *, exclude_kanji: Set[str] | None = None
) -> Tuple[List[int], List[int], List[int]]:
    strokes: List[int] = []
    freqs: List[int] = []
    jlpt_levels: List[int] = []

    # Unique kanji seulement
    exclude = exclude_kanji or set()
    base = _standalone_kanji_set(text)
    unique_kanji: Set[str] = {ch for ch in base if (ch in kanji_meta) and (ch not in exclude)}

    for ch in unique_kanji:
        meta = kanji_meta.get(ch)
        if not meta:
            continue

        strokes_val = meta.get("strokes")
        if isinstance(strokes_val, int):
            strokes.append(strokes_val)

        freq_val = meta.get("frequency")
        if isinstance(freq_val, int):
            freqs.append(freq_val)

        jlpt_val = meta.get("jlpt")
        if isinstance(jlpt_val, int):
            jlpt_levels.append(jlpt_val)

    return strokes, freqs, jlpt_levels


def _kanji_complexity_features(
    text: str, kanji_meta: Dict[str, Dict[str, int]], *, exclude_kanji: Set[str] | None = None
) -> Dict[str, float]:
    strokes, freqs, jlpt_levels = _collect_kanji_meta_lists(text, kanji_meta, exclude_kanji=exclude_kanji)

    if not strokes and not freqs and not jlpt_levels:
        return {
            "kanji_strokes_avg": 0.0,
            "kanji_strokes_max": 0.0,
            "kanji_freq_avg": 0.0,
            "kanji_freq_min": 0.0,
            "kanji_jlpt_avg_num": 0.0,
        }

    def _avg(xs: List[int]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "kanji_strokes_avg": _avg(strokes) if strokes else 0.0,
        "kanji_strokes_max": float(max(strokes)) if strokes else 0.0,
        "kanji_freq_avg": _avg(freqs) if freqs else 0.0,
        "kanji_freq_min": float(min(freqs)) if freqs else 0.0,
        "kanji_jlpt_avg_num": _avg(jlpt_levels) if jlpt_levels else 0.0,
    }


def _grammar_features(text: str) -> Dict[str, float]:
    # Préparer une fois (avec cache) pour les performances.
    pairs = _get_grammar_pairs()

    counts = _count_presence_pruned(text, pairs)

    features = {f"grammar_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["grammar_total"] = float(sum(counts.values()))
    features.update(_level_ratios("grammar", counts, int(features["grammar_total"])))
    features["grammar_max_level_num"] = _max_level_num(counts)
    return features


def extract_match_trace(text: str, *, max_per_level: int = 25) -> Dict[str, Dict[str, List[str]]]:
    """Renvoie les tokens/patterns matchés (debug / vérification).

    N'affecte pas l'entraînement : sert à comprendre pourquoi un texte est jugé
    avancé (N2/N1) et à repérer des erreurs de dictionnaire/matching.
    """

    text_norm = _normalize_text(text)
    vocab_pairs = _get_vocab_pairs()
    _, vocab_by_level = _match_vocab_with_inference(text_norm, vocab_pairs)

    grammar_pairs = _get_grammar_pairs()
    _, grammar_by_level = _match_presence_pruned(text_norm, grammar_pairs)

    # Tronquer les listes pour garder une sortie lisible.
    for lvl in LEVELS:
        vocab_by_level[lvl] = vocab_by_level[lvl][: max(0, int(max_per_level))]
        grammar_by_level[lvl] = grammar_by_level[lvl][: max(0, int(max_per_level))]

    return {
        "vocab": vocab_by_level,
        "grammar": grammar_by_level,
    }


def extract_features(text: str) -> Dict[str, float]:
    text_norm = _normalize_text(text)

    kanji_json = _load_json(KANJI_JSON)
    kanji_meta = _load_json(KANJI_META_JSON) if KANJI_META_JSON.exists() else {}

    features: Dict[str, float] = {}
    features.update(_char_stats(text_norm))
    features.update(_sentence_stats(text_norm))
    features.update(_particle_features(text_norm))

    vocab_pairs = _get_vocab_pairs()
    features.update(_vocab_features(text_norm, vocab_pairs))

    vocab_tokens = _extract_vocab_matched_tokens(text_norm, vocab_pairs)
    # Heuristique vocab d'abord : si un kanji est déjà couvert par un token vocab JLPT,
    # on ne le recompte pas en difficulté kanji isolé.
    covered_kanji = _kanji_set_from_tokens(vocab_tokens, allow_levels=set(LEVELS))

    features.update(_kanji_features(text_norm, kanji_json, exclude_kanji=covered_kanji))
    features.update(_kanji_complexity_features(text_norm, kanji_meta, exclude_kanji=covered_kanji))
    features.update(_grammar_features(text_norm))
    return features


if __name__ == "__main__":
    # Test rapide
    sample = "今日はいい天気です。"
    feats = extract_features(sample)
    for k in sorted(feats.keys()):
        print(f"{k}: {feats[k]}")
