"""Narrow, reviewable conversation-handling decisions (not ML predictions).

These detectors are not a general-purpose mixed-intent model. They address a
small class of supported, auditable questions without making account claims.
"""
import re

# Recognize an explicitly non-banking proposition alongside banking terminology.
# General named entities alone are NOT enough to reject banking complaints.
SPORTS_PROPOSITION = re.compile(
    r"\b(?:team india|cricket|football|world cup|ipl|match|tournament)\b"
    r".{0,90}\b(?:win|won|winning|lose|loses|lost|losing|defeat|score|result|beat)\b|"
    r"\b(?:win|won|winning|lose|loses|lost|losing|defeat|score|result|beat)\b"
    r".{0,90}\b(?:team india|cricket|football|world cup|ipl|match|tournament)\b", re.I)
FINANCIAL_ISSUE = re.compile(
    r"\b(?:debit card|credit card|bank card|bank(?:ing)?|bank account|atm|"
    r"card payment|payment failure|card failure|transaction|refund|transfer)\b", re.I)
MISSING_MONEY = re.compile(
    r"\b(?:money|funds|amount|balance|savings)\b.{0,45}"
    r"\b(?:missing|disappeared|gone)\b.{0,45}\b(?:account|balance|bank)\b|"
    r"\b(?:money|funds|amount|balance|savings)\b.{0,45}"
    r"\b(?:missing|disappeared|gone)\b|"
    r"\b(?:missing|disappeared|gone)\b.{0,40}\b(?:money|funds|balance)\b", re.I)


def conversation_decision(question):
    """Returns a conservative clarification kind, or None.

    Explicit fraud, security and account-verification rules must take precedence
    in the caller. An ambiguous missing balance is NOT proof of fraud.
    """
    q=question.strip()
    if SPORTS_PROPOSITION.search(q) and FINANCIAL_ISSUE.search(q):
        return 'mixed_domain'
    if MISSING_MONEY.search(q) and re.search(r'\b(?:account|bank|balance|funds)\b',q,re.I):
        return 'missing_funds_ambiguous'
    return None


def clarification_response(kind):
    if kind=='mixed_domain':
        return ("Your message combines a non-banking topic with a possible banking issue. "
                "Could you clarify what happened with your card, payment, or account? "
                "I can help with banking-support questions, but cannot determine a banking "
                "problem from the sports question. No account action was performed. "
                "Human review is required.")
    if kind=='missing_funds_ambiguous':
        return ("I understand that money appears to be missing from your account. "
                "Please check the transaction history in your bank's official app or other "
                "verified channel. If you see an unfamiliar or unauthorized transaction, "
                "contact your bank's verified fraud-support channel promptly. "
                "Is the amount missing from your balance, or do you see a transaction you "
                "don't recognize? Do not share passwords, PINs, CVVs, or OTPs. "
                "This demonstration cannot verify your balance or transactions. "
                "No account action was performed. Human review is required.")
    raise ValueError('Unknown clarification kind')
