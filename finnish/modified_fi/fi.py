"""
Copyright 2019-2023 Antti Ajanki <antti.ajanki@iki.fi>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

----------------

This is a modified version of "fi.py" file, originally published as 
a part of the "Experimental Finnish language model for spaCy" package 
(https://github.com/aajanki/spacy-fi).

Edits made by Heidi Huhtanen 2024.

----------------

"""

import re
import srsly
import os
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Union
from spacy import util
from spacy.errors import Errors
from spacy.lang.fi import Finnish, FinnishDefaults
from spacy.language import Language
from spacy.lookups import Lookups, load_lookups
from spacy.pipeline.pipe import Pipe
from spacy.scorer import Scorer
from spacy.symbols import ADJ, ADP, ADV, AUX, CCONJ, INTJ, NOUN, NUM, PROPN
from spacy.symbols import PRON, PUNCT, SCONJ, SPACE, SYM, VERB, X
from spacy.symbols import conj, obj
from spacy.tokens import Doc, Span, Token
from spacy.training import Example, validate_examples
from spacy.util import SimpleFrozenList
from spacy.vocab import Vocab
from voikko import libvoikko


class VoikkoLemmatizer(Pipe):
    """Pipeline component that assigns lemmas to Docs.

    Lemmatization is done by libvoikko.
    """
    compound_re = re.compile(r"\+(\w+)(?:\(\+?[\w=]+\))?")
    minen_re = re.compile(r"\b(\w+)\[Tn4\]mi")
    ny_re = re.compile(r"\[X\]\[\w+\]\[Ny\](\w+)")
    roman_numeral_structure_re = re.compile(r"=j+|=q+")
    voikko_classes_by_pos = {
        ADJ:   frozenset(["laatusana", "nimisana_laatusana"]),
        ADP:   frozenset(["nimisana", "seikkasana", "suhdesana"]),
        ADV:   frozenset(["seikkasana"]),
        AUX:   frozenset(["teonsana", "kieltosana"]),
        CCONJ: frozenset(["sidesana"]),
        INTJ:  frozenset(["huudahdussana"]),
        NOUN:  frozenset(["nimisana", "nimisana_laatusana", "lyhenne"]),
        NUM:   frozenset(["lukusana"]),
        PRON:  frozenset(["asemosana", "nimisana", "nimisana_laatusana"]),
        PROPN: frozenset(["nimi", "etunimi", "sukunimi", "paikannimi"]),
        SCONJ: frozenset(["sidesana"]),
        VERB:  frozenset([]),  # Would be "teonsana" except that
                               # MINEN-infinitives are treated as nouns.
                               # See _analysis_has_compatible_pos()
        SYM:   frozenset([]),
        X:     frozenset([])
    }
    possessive_suffixes = {
        "1s": ["ni"],
        "2s": ["si"],
        "1p": ["mme"],
        "2p": ["nne"],
        "3": ["nsa", "nsä", "an", "en", "in" "on", "un", "yn", "än", "ön"],
    }

    def __init__(
            self,
            vocab: Vocab,
            name: str = "lemmatizer",
            *,
            overwrite_lemma: bool = False,
    ) -> None:
        super().__init__()
        # editoidut
        VOIKKO_PATH = "./voikko-dictionary/5/"

        self.name = name
        self.vocab = vocab
        self.voikko = libvoikko.Voikko("fi-x-morpho", path=VOIKKO_PATH)
        
        # Originaalit
        #self.name = name
        #self.vocab = vocab
        #self.voikko = libvoikko.Voikko("fi")
        self.lookups = Lookups()
        self.overwrite_lemma = overwrite_lemma
        self.nsubj_labels = [vocab.strings.add(x) for x in ["nsubj", "nsubj:cop"]]

    def __call__(self, doc: Doc) -> Doc:
        error_handler = self.get_error_handler()
        try:
            for token in doc:
                if token.pos in (PUNCT, SPACE):
                    if self.overwrite_lemma or token.lemma == 0:
                        token.lemma = token.orth
                else:
                    if self.overwrite_lemma or token.lemma == 0:
                        analysis = self._analyze(token)
                        token.lemma_ = self.lemmatize(token, analysis)
            return doc
        except Exception as e:
            error_handler(self.name, self, [doc], e)

    def initialize(
        self,
        get_examples: Optional[Callable[[], Iterable[Example]]] = None,
        *,
        nlp: Optional[Language] = None,
        lookups: Optional[Lookups] = None,
    ):
        """Initialize the lemmatizer and load in data.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        lookups (Lookups): The lookups object containing the (optional) tables
            such as "lemma_exc". Defaults to None.
        """
        if lookups is None:
            lookups = load_lookups(lang=self.vocab.lang, tables=["lemma_exc"])
        self.lookups = lookups

    def lemmatize(self, token: Token, analysis: dict) -> str:
        cached_lower = None
        exc_table = self.lookups.get_table("lemma_exc", {}).get(token.pos)
        if exc_table is not None:
            cached_lower = token.orth_.lower()
            exc = exc_table.get(cached_lower)
            if exc:
                return exc

        # Some exceptions to Voikko's lemmatization algorithm to
        # better match UD lemmas
        if token.pos in (AUX, VERB) and "PARTICIPLE" in analysis:
            return self._participle_lemma(analysis)
        elif token.pos == NOUN and analysis.get("MOOD") == "MINEN-infinitive":
            return self._minen_noun_lemma(analysis)
        elif token.pos in (NOUN, NUM, PROPN) and ':' in token.orth_:
            # Lemma of an inflected abbreviation: BBC:n -> BCC, EU:ssa -> EU
            colon_i = token.orth_.find(":")
            return token.orth_[:colon_i]
        elif token.pos == ADV:
            cached_lower = cached_lower or token.orth_.lower()
            return self._adv_lemma(analysis, cached_lower)
        elif token.pos == ADP:
            return cached_lower or token.orth_.lower()
        elif "BASEFORM" not in analysis:
            if token.pos in (PROPN, INTJ, SYM, X):
                return token.orth_
            else:
                return cached_lower or token.orth_.lower()
        else:
            return analysis["BASEFORM"]

    def _analyze(self, token):
        orth = token.orth_
        if '-' in orth:
            # Analyze only the head token on hyphenated compound
            # words.
            parts = [x for x in orth.rsplit('-', 1) if x]
            if parts:
                orth = parts[-1]
        else:
            parts = [orth]

        analyses = self.voikko.analyze(orth)
        analysis = self._disambiguate_analyses(token, analyses)
        if len(parts) > 1 and "BASEFORM" in analysis:
            analysis["BASEFORM"] = parts[0] + "-" + analysis["BASEFORM"]
        return analysis

    def _disambiguate_analyses(self, token, analyses):
        matching_pos = [x for x in analyses if self._analysis_has_compatible_pos(token, x)]
        if matching_pos:
            analyses = matching_pos

        if len(analyses) > 1:
            # Disambiguate among multiple possible analyses

            if token.pos == NUM:
                # For numbers like 1,5 prefer the analysis without
                # SIJAMUOTO and NUMBER because UD doesn't have them,
                # either.
                analyses = [x for x in analyses if "SIJAMUOTO" not in x] or analyses

                # Prefer uppercase Roman numerals
                if all(self.roman_numeral_structure_re.fullmatch(x.get("STRUCTURE", ""))
                       for x in analyses):
                    analyses = [x for x in analyses if "j" in x.get("STRUCTURE")]

            elif token.pos in (NOUN, PRON) and \
                 ((token.dep in self.nsubj_labels) or
                  (token.dep == conj and token.head.dep in self.nsubj_labels)):
                # Subject is usually nominative, genetive or partitive
                analyses = [
                    x for x in analyses
                    if x.get("SIJAMUOTO") in ["nimento", "omanto", "osanto"]
                ] or analyses

            elif token.pos in (NOUN, PRON) and \
                 (token.dep == obj or (token.dep == conj and token.head.dep == obj)):
                # Object is usually partitive, accusative or genetive
                analyses = [
                    x for x in analyses
                    if x.get("SIJAMUOTO") in ("kohdanto", "omanto", "osanto")
                ] or analyses

            elif token.pos == NOUN:
                # Prefer non-compound words.
                # e.g. "asemassa" will be lemmatized as "asema", not "ase#massa"
                analyses = sorted(analyses, key=self._is_compound_word)

        if analyses:
            return analyses[0]
        else:
            return {}

    def _analysis_has_compatible_pos(self, token, analysis):
        if "CLASS" not in analysis:
            return False

        tpos = token.pos
        vclass = analysis["CLASS"]

        return (
            vclass in self.voikko_classes_by_pos[tpos]

            or

            (tpos == NOUN and vclass == "teonsana" and analysis.get("MOOD") == "MINEN-infinitive")

            or

            (tpos == VERB and (
                (vclass == "teonsana" and analysis.get("MOOD") != "MINEN-infinitive") or
                ("PARTICIPLE" in analysis and (
                    # agent participle
                    (vclass == "nimisana" and analysis["PARTICIPLE"] == "agent") or
                    # VA, NUT and TU participles
                    (vclass == "laatusana" and analysis["PARTICIPLE"] in (
                        "past_active", "past_passive", "present_active", "present_passive"))))))

            or

            (tpos == ADV and
             vclass in ("laatusana", "lukusana") and
             analysis.get("SIJAMUOTO") == "kerrontosti")
        )

    def _minen_noun_lemma(self, analysis):
        fstoutput = analysis.get("FSTOUTPUT")
        ny_match = self.ny_re.search(fstoutput)
        if ny_match:
            return ny_match.group(1)

        fst_match = self.minen_re.search(fstoutput)
        if not fst_match:
            return None

        stem = fst_match.group(1)
        compounds = self.compound_re.findall(analysis.get("WORDBASES"))
        if len(compounds) > 1:
            return "".join(compounds[:-1]) + stem + "minen"
        else:
            return stem + "minen"

    def _participle_lemma(self, analysis):
        wordbases = analysis.get("WORDBASES")
        num_bases = max(analysis.get("STRUCTURE", "").count("="), 1)

        i = 0
        forms = []
        for base in re.finditer(r"\+([^+]+)", wordbases):
            full_form = base.group(1)
            parentheses_match = re.search(r"(.+)\((.+)\)", full_form)
            if parentheses_match:
                k = parentheses_match.group(2).count("=") + 1
                if i < num_bases - k:
                    form = parentheses_match.group(1)
                else:
                    form = parentheses_match.group(2)
            else:
                form = full_form

            split = form.split("=")
            forms.extend(split)
            i += len(split)

            if i >= num_bases:
                break

        return ''.join(forms)

    def _adv_lemma(self, analysis, word):
        focus = analysis.get("FOCUS")
        has_kysymysliite = "KYSYMYSLIITE" in analysis

        if focus and has_kysymysliite:
            k = 2
        elif focus or has_kysymysliite:
            k = 1
        else:
            k = 0
        for _ in range(k):
            if focus and word.endswith(focus):
                word = word[:-len(focus)]
            elif has_kysymysliite and (word.endswith("ko") or word.endswith("kö")):
                word = word[:-2]

        if "POSSESSIVE" in analysis and analysis["POSSESSIVE"] != "3":
            # "kanssani", "vierestäni"
            word = self._remove_possessive_suffix(word, analysis)

        return word

    def _remove_possessive_suffix(self, word, analysis):
        """Removes possessive suffix from the word.

        Example: "kanssamme" -> "kanssa"
        """
        suffixes = self.possessive_suffixes[analysis["POSSESSIVE"]]
        suffix = next((s for s in suffixes if word.endswith(s)), None)
        if not suffix:
            return word

        word = word[:-len(suffix)]
        if analysis.get("SIJAMUOTO") == "tulento" and word.endswith("e"):
            # onne+kse+mme -> onne+ksi
            word = word[:-1] + "i"
        elif analysis.get("SIJAMUOTO") == "sisatulento":
            # lapsee+ni -> lapseen
            word = word + "n"
        elif suffix.startswith('n') and analysis.get("BASEFORM").endswith("n"):
            # mukaa+ni -> mukaan
            word = word + "n"

        return word

    def _is_compound_word(self, analysis):
        structure = analysis.get('STRUCTURE', '')
        return structure.count('=') > 1

    def score(self, examples, **kwargs):
        validate_examples(examples, "VoikkoLemmatizer.score")
        results = {}
        results.update(Scorer.score_token_attr(examples, "lemma", **kwargs))
        return results

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ):
        serialize = {"lookups": lambda p: self.lookups.to_disk(p)}
        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "VoikkoLemmatizer":
        deserialize = {"lookups": lambda p: self.lookups.from_disk(p)}
        util.from_disk(path, deserialize, exclude)
        return self

    def to_bytes(self, *, exclude: Iterable[str] = SimpleFrozenList()) -> bytes:
        serialize = {"lookups": self.lookups.to_bytes}
        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "VoikkoLemmatizer":
        deserialize = {"lookups": lambda b: self.lookups.from_bytes(b)}
        util.from_bytes(bytes_data, deserialize, exclude)
        return self


@Finnish.factory(
    "voikko_lemmatizer",
    assigns=["token.lemma"],
    requires=[],
    default_score_weights={"lemma_acc": 0.0},
)
def make_voikko_lemmatizer(
    nlp: Language,
    name: str,
    overwrite_lemma: bool = False
):
    return VoikkoLemmatizer(nlp.vocab, name, overwrite_lemma=overwrite_lemma)


def noun_chunks(doclike: Union[Doc, Span]) -> Iterator[Span]:
    """Detect base noun phrases from a dependency parse. Works on both Doc and Span."""
    labels = [
        "nsubj",
        "nsubj:cop",
        "obj",
        "obl",
        "ROOT",
    ]
    extend_labels = [
        "advmod",
        "amod",
        "appos",
        "case",
        "compound",
        "compound:nn",
        "flat:name",
        "nmod",
        "nmod:gobj",
        "nmod:gsubj",
        "nmod:poss",
        "nummod",
    ]

    def potential_np_head(word):
        # TODO: PRON handling is inconsistent. Should some pronouns
        # (indefinite?, personal?) be considered part of a noun chunk?
        return word.pos in (NOUN, PROPN) and (word.dep in np_deps or word.head.pos == PRON)

    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)

    np_deps = [doc.vocab.strings[label] for label in labels]
    extend_deps = [doc.vocab.strings[label] for label in extend_labels]
    np_label = doc.vocab.strings.add("NP")
    conj_label = doc.vocab.strings.add("conj")

    rbracket = 0
    prev_end = -1
    for i, word in enumerate(doclike):
        if i < rbracket:
            continue

        # Is this a potential independent NP head or coordinated with
        # a NOUN that is itself an independent NP head?
        #
        # e.g. "Terveyden ja hyvinvoinnin laitos"
        if potential_np_head(word) or (word.dep == conj_label and potential_np_head(word.head)):
            # Try to extend to the left to include adjective/num
            # modifiers, compound words etc.
            lbracket = word.i
            for ldep in word.lefts:
                if ldep.pos in (NOUN, PROPN, NUM, ADJ) and ldep.dep in extend_deps:
                    lbracket = ldep.left_edge.i
                    break

            # Prevent nested chunks from being produced
            if lbracket <= prev_end:
                continue

            rbracket = word.i
            # Try to extend the span to the right to capture close
            # appositions and noun modifiers
            for rdep in word.rights:
                if rdep.dep in extend_deps:
                    rbracket = rdep.i
                    for j in range(rdep.i + 1, rdep.right_edge.i + 1):
                        if doc[j].dep in extend_deps:
                            rbracket = j
            prev_end = rbracket

            yield lbracket, rbracket + 1, np_label


@util.registry.misc("spacyfi.read_lookups_from_json.v1")
def create_lookups_from_json_reader(path: Path) -> Lookups:
    lookups = Lookups()
    for p in path.glob("*.json"):
        table_name = p.stem
        data = srsly.read_json(p)
        lookups.add_table(table_name, data)
    return lookups


class FinnishExDefaults(FinnishDefaults):
    syntax_iterators = {"noun_chunks": noun_chunks}


@util.registry.languages("fi")
class FinnishExtended(Language):
    """Extends the default Finnish language class with syntax iterators."""
    lang = 'fi'
    Defaults = FinnishExDefaults
