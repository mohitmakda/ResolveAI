"""Independent, conservative input and evidence decisions for the fictional banking copilot.

Scores and thresholds are heuristics, not calibrated correctness probabilities.
This module has no Streamlit/Ollama dependency and can be regression-tested separately.
"""
import re
from functools import lru_cache

BANKING_ANCHORS = [
    'My card is not working', 'Someone took my money', 'I cannot sign in to mobile banking',
    'My bank account was accessed by someone else', 'I cannot log into my account',
    'I entered the wrong credentials repeatedly and the login is locked',
    'My card was stolen', 'I need to replace my debit card',
    'I sent a bank transfer but the recipient has not received it',
    'My ATM withdrawal failed and my account was debited',
    'I do not recognize this card transaction', 'My banking password was shared with another person',
    'How do I update my personal details with my bank', 'My bank card payment was declined',
    'My refund has not appeared in my account', 'I was charged twice for a purchase',
    'My account is restricted', 'My credit card was never delivered',
    'How can I dispute a transaction', 'I cannot access my money',
    'I forgot my passcode', 'How can I check my balance',
    'My cash withdrawal was not received', 'I want to know the status of my reversal',
    'My funds disappeared', 'Someone is withdrawing my funds',
]

# These training prompts are independent of the final regression evaluation set.
OOD_TRAIN = [
    'Explain the weather forecast tomorrow', 'What temperature is it outside',
    'Tell a funny joke', 'Write a poem about autumn', 'Create JavaScript code to sort numbers',
    'Recommend a TV series', 'Who won the football championship',
    'Who is the president of Brazil', 'Capital city of Australia',
    'Recipe for chocolate cake', 'Explain quantum physics', 'Translate a Spanish sentence',
    'Plan my holiday to Italy', 'What is the nearest train station',
    'Play a music playlist', 'What is the best camera to buy',
    'Help debug my Python program', 'Explain a sorting algorithm',
    'How do I water a houseplant', 'Solve this calculus equation',
    'Summarize an ancient history chapter', 'Recommend comedy movies',
    'How is the cricket score today', 'Find a good restaurant',
    'How many planets are in the solar system', 'Explain cloud formation',
    'Draft an email to my teacher', 'What does this English word mean',
    'Generate a picture of a mountain', 'Give me a workout plan',
]

@lru_cache(maxsize=1)
def _domain_model():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from core import EXAMPLES
    banking = [p['text'] for p in EXAMPLES] + BANKING_ANCHORS
    texts = banking + OOD_TRAIN
    y = [1] * len(banking) + [0] * len(OOD_TRAIN)
    model = Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=2.0)),
    ])
    model.fit(texts, y)
    return model

# Explicit banking context is a *supplement* to the trained domain classifier,
# not the only criterion. Avoid overly generic anchors such as "money" alone.
BANKING_CONTEXT = re.compile(
    r'\b(bank(?:ing)?|account|atm|card|debit|credit|transaction|transfer|'
    r'payment|withdrawal|refund|reversal|passcode|otp|cvv|pin|'
    r'fraud|chargeback|beneficiary|cash|salary deposit|balance|'
    r'funds|money|login|log in|sign in)\b', re.I)

# Definite non-support subjects must not be passed just because the prompt says "bank".
META_OOD = re.compile(
    r'\b(?:write|generate|create|implement|debug)\b.{0,35}'
    r'\b(?:code|python|javascript|program|script|poem|joke|story)\b', re.I)


def check_domain(question):
    """Return a domain assessment; no intent classification for rejected prompts."""
    q = question.strip()
    if not q:
        return {'in_domain': False, 'score': 0.0, 'reason': 'Empty message'}
    if META_OOD.search(q) and not re.search(r'\b(?:banking|bank|account|card|transaction)\s+(?:app|login|issue|support|problem)\b', q, re.I):
        return {'in_domain': False, 'score': 0.0, 'reason': 'Non-support task'}
    score = float(_domain_model().predict_proba([q])[0][1])
    has_context = bool(BANKING_CONTEXT.search(q))
    # Similarity-trained binary signal + broad context: never treat probability as calibrated.
    # Very short banking-language complaints remain eligible for further evaluation.
    accepted = score >= 0.51 or (score >= 0.30 and has_context)
    return {'in_domain': accepted, 'score': round(score, 4),
            'reason': 'Banking support domain' if accepted else 'Outside banking-support domain'}


CREDENTIAL_EXPOSURE = re.compile(
    r'\b(?:shared|sent|gave|told|disclosed|posted|leaked|revealed)\b.{0,70}'
    r'\b(?:password|passcode|pin|otp|cvv|login credentials?)\b|'
    r'\b(?:password|passcode|pin|otp|cvv)\b.{0,45}\b(?:shared|leaked|exposed)\b', re.I)
ACCOUNT_COMPROMISE = re.compile(
    r'\b(?:someone|somebody|stranger|another person|unknown person|hacker)\b'
    r'.{0,80}\b(?:accessed|logged into|got into|entered|took over|used|hacked)\b'
    r'.{0,70}\b(?:account|banking|login|funds)\b|'
    r'\b(?:account|banking)\b.{0,75}\b(?:without (?:my )?(?:permission|consent)|compromised|hacked)\b', re.I)
UNRECOGNIZED_TRANSACTION = re.compile(
    r'\b(?:don.t|do not|cannot|can.t)\s+(?:recogni[sz]e|remember)\b.{0,65}'
    r'\b(?:transaction|charge|payment|withdrawal|debit)\b|'
    r'\b(?:transaction|charge|payment|withdrawal|debit)\b.{0,65}'
    r'\b(?:don.t|do not|cannot|can.t)\s+(?:recogni[sz]e|remember)\b', re.I)
INJECTION = re.compile(
    r'\b(?:ignore|override|disregard|forget)\b.{0,65}'
    r'\b(?:previous|above|system|developer|safety|instructions?|rules?)\b|'
    r'\b(?:reveal|print|show)\b.{0,50}\b(?:system prompt|hidden prompt|developer prompt)\b', re.I)


def input_risk(question):
    if INJECTION.search(question):
        return 'prompt_injection'
    if CREDENTIAL_EXPOSURE.search(question):
        return 'credentials_exposed'
    if UNAUTHORIZED_FUNDS.search(question):
        return 'unauthorized_funds'
    if ACCOUNT_COMPROMISE.search(question):
        return 'account_compromise'
    if UNRECOGNIZED_TRANSACTION.search(question):
        return 'unrecognized_transaction'
    return None


# Distinguish stolen funds from unauthorized login and from card disputes.
UNAUTHORIZED_FUNDS = re.compile(
    r"\b(?:someone|somebody|stranger|unknown person)\b.{0,60}\b(?:took|stole|withdrawn|withdrew|transferred|removed|debited)\b.{0,60}\b(?:money|funds|account|balance)\b|"
    r"\b(?:money|funds)\b.{0,45}\b(?:taken|stolen|withdrawn|transferred)\b.{0,50}\bwithout (?:my )?(?:permission|consent|authori[sz]ation)\b|"
    r"\bunauthori[sz]ed\b.{0,35}\b(?:transfer|withdrawal|debit|bank transaction)\b", re.I)

def risk_response(risk):
    if risk == 'prompt_injection':
        return ('I can assist only with banking-support complaints. Instructions to bypass '
                'the assistant’s safety or policy rules are not processed. Human review is required.')
    if risk == 'credentials_exposed':
        return ('You may have disclosed a sensitive banking credential. Contact your bank immediately '
                'through a verified official channel for security guidance and appropriate credential '
                'replacement. Do not share passwords, PINs, CVVs, or OTPs here. No account action '
                'has been performed. Human review is required.')
    if risk == 'account_compromise':
        return ('Your message suggests possible unauthorized account access. Contact your bank '
                'immediately using a verified official security channel for a security review. '
                'Do not share credentials. No account has been accessed or secured by this demo. '
                'Human review is required.')
    if risk == 'unauthorized_funds':
        return ('Your message indicates a possible unauthorized withdrawal or transfer. Contact your bank promptly '
                'through a verified official fraud channel to report it and request an investigation. '
                'No recovery or reversal is confirmed; no account action was taken. Human review is required.')
    if risk == 'unrecognized_transaction':
        return ('This may be an unauthorized transaction. Contact verified official bank fraud '
                'support promptly and request a review. A refund or reversal is not confirmed. '
                'No account action was taken. Human review is required.')
    return None


def evidence_decision(question, docs, category, semantic=False):
    """Return relevant-policy and answerable flags separately.

    Structural coverage is required; account-status questions are never answerable here.
    """
    from core import policy_supported
    relevant = policy_supported(docs, category, semantic=semantic)
    # This gate is intentionally independent of retrieval and model prediction.
    from app_guardrails import requires_account_verification
    private = requires_account_verification(question)
    return {'relevant': relevant, 'answerable': bool(relevant and not private),
            'private_records': private,
            'reason': ('Private bank records unavailable' if private else
                       'Relevant fictional policy found' if relevant else
                       'Insufficient applicable policy evidence')}


def policy_routing_override(question, urgency):
    """A few high-precision complaint patterns outrank ambiguous classifier labels.

    Returns a policy category, not an asserted BANKING77 classification.
    """
    if urgency.get('suggested_category'):
        return urgency['suggested_category']
    q = question.lower()
    patterns = [
        ('card_payment_issue', r'\b(?:card|contactless)\b.{0,35}\b(?:not working|doesn.t work|declined|failed|rejected)\b|\b(?:payment|purchase)\b.{0,35}\b(?:declined|rejected)\b'),
        ('refund_guidance', r'\b(?:refund|reversal|chargeback)\b'),
        ('login_issue', r'\b(?:login|log in|sign in|passcode|password|credentials?)\b.{0,35}\b(?:locked|incorrect|wrong|forgot|trouble|failed|cannot|can.t)\b|\b(?:locked|incorrect|wrong|forgot|trouble|failed|cannot|can.t)\b.{0,35}\b(?:login|log in|sign in|passcode|password|credentials?)\b'),
        ('account_restriction', r'\b(?:account)\b.{0,35}\b(?:restricted|frozen|suspended)\b'),
        ('card_delivery', r'\b(?:card)\b.{0,35}\b(?:arrival|arrived|delivery|delivered|shipping|shipped)\b'),
        ('atm_issue', r'\b(?:atm|cash withdrawal)\b.{0,55}\b(?:failed|no cash|debited|did not|didn.t|missing|not received)\b'),
        ('transfer_issue', r'\b(?:transfer|beneficiary)\b.{0,45}\b(?:failed|missing|not received|pending|delayed)\b'),
    ]
    for category, pattern in patterns:
        if re.search(pattern, q, re.I):
            return category
    return None


def resolve_intent(question, model_label, confidence, urgency, risk=None):
    """Human-readable issue interpretation; never masquerades as ML classifier output.

    Deterministic overrides are limited to high-specificity cases, and low-confidence
    ambiguous requests remain visible as uncertain. Return provenance for auditing.
    """
    if risk == 'unauthorized_funds':
        return {'intent': 'Unauthorized account transaction', 'category': 'unauthorized_transaction',
                'source': 'independent fraud safety rule'}
    if risk == 'unrecognized_transaction':
        return {'intent': 'Unrecognized card transaction', 'category': 'card_dispute',
                'source': 'independent transaction safety rule'}
    if risk == 'credentials_exposed':
        return {'intent': 'Exposed banking credential', 'category': 'credential_security',
                'source': 'independent credential safety rule'}
    if risk == 'account_compromise':
        return {'intent': 'Possible account compromise', 'category': 'login_issue',
                'source': 'independent account safety rule'}
    overrides = [
        ('Declined card payment', 'card_payment_issue', r'\b(?:card|debit|credit|contactless)\b.{0,50}\b(?:payment|purchase|transaction)\b.{0,30}\b(?:declined|rejected|failed)\b|\b(?:card|debit|credit)\b.{0,50}\b(?:declined|rejected)\b|\b(?:payment|purchase)\b.{0,30}\b(?:declined|rejected)\b'),
        ('Lost or stolen card', 'lost_card', r'\b(?:lost|stolen|missing)\b.{0,35}\bcard\b|\bcard\b.{0,35}\b(?:lost|stolen|missing)\b'),
        ('ATM withdrawal issue', 'atm_issue', r'\b(?:atm|cash withdrawal)\b.{0,55}\b(?:failed|no cash|debited|not received|did not|didn.t)\b'),
        ('Login or access issue', 'login_issue', r'\b(?:cannot|can.t|unable|forgot|locked|wrong|incorrect)\b.{0,45}\b(?:log in|login|sign in|passcode|password|credentials?)\b|\b(?:log in|login|sign in|passcode|password)\b.{0,45}\b(?:locked|failed|forgot|not working)\b'),
    ]
    for intent, category, pattern in overrides:
        if re.search(pattern, question, re.I):
            return {'intent': intent, 'category': category, 'source': 'high-precision issue rule'}
    mapped = policy_routing_override(question, urgency)
    if mapped:
        return {'intent': mapped.replace('_', ' ').title(), 'category': mapped, 'source': 'policy routing rule'}
    # With 77 fine-grained classes, a weak top score cannot justify a precise
    # complaint type. Abstain from choosing a policy instead of inventing one.
    # This heuristic cutoff requires future validation on unseen banking queries.
    if confidence < 0.30:
        return {'intent': 'Banking issue — needs clarification', 'category': None,
                'source': 'low-confidence ML abstention (heuristic)', 'uncertain': True}
    from core import policy_hint
    mapped = policy_hint(model_label, urgency)
    return {'intent': model_label.replace('_', ' ').title(), 'category': mapped,
            'source': 'ML prediction (uncalibrated)', 'uncertain': confidence < 0.5}
