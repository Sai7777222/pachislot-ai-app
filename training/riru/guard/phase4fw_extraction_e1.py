"""Phase4FW Stage A / E1: deterministic atomic claim extraction。
句読点・接続詞・列挙構造を使った分割のみ。巨大regexは使わない(単純な区切り文字のみ)。"""
from __future__ import annotations
import re

# 文を区切る記号(句点、読点+接続語、「一方」「また」等の対比・並列マーカー)
SPLIT_MARKERS = re.compile(
    r"(?<=[。])|(?<=、)(?=一方|また|さらに|なお|逆に|それに対して|対して)"
)
CONTRAST_MARKERS = re.compile(r"一方[、,]?|それに対して[、,]?|逆に[、,]?")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SPLIT_MARKERS.split(text) if p and p.strip()]
    return parts


def split_clauses(sentence: str) -> list[str]:
    """一文中に複数対象への言及がある場合、対比マーカーや並列助詞でさらに分割を試みる。"""
    # 対比マーカーで分割(「AはX、一方BはY」型)
    if CONTRAST_MARKERS.search(sentence):
        clauses = CONTRAST_MARKERS.split(sentence)
        return [c.strip() for c in clauses if c and c.strip()]
    return [sentence]


def extract_claims_e1(text: str) -> list[dict]:
    """deterministic segmentation: 文単位→対比節単位に分割し、各節を1 atomic claim候補とする。
    subject推定は簡易的に節冒頭の名詞句(最初の「は」「が」までの部分)を採用する。"""
    claims = []
    idx = 0
    for sent in split_sentences(text):
        for clause in split_clauses(sent):
            m = re.match(r"^([^\sはがのをにでと、。]+(?:[のと][^\sはがのをにでと、。]+)?)[はが]", clause)
            subject_guess = m.group(1) if m else None
            claims.append({"claim_id": idx, "raw_text": clause, "subject_guess": subject_guess})
            idx += 1
    return claims


if __name__ == "__main__":
    sample = "GGはGG準備中から始まって、GG本前兆→GG本前兆→GG本前兆→GG本当選という流れで進むんだ。GG本前兆は「×・?・?」の順番で、GG本当選は「×・?・×」の順番になるよ。一方、SGGはGG本前兆の後にSGGゾーンが始まるんだ。"
    for c in extract_claims_e1(sample):
        print(c)
