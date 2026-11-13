"""Conservative, offline request-purpose gate (not an LLM or a calibrated model).

It aims to prevent keyword-triggered false banking instructions. Uncertainty returns
clarification, never a fabricated banking incident. Security guardrails remain
independent and take precedence in the Streamlit caller.
"""
from functools import lru_cache
import re

POSITIVE = ['My payment was rejected at checkout', 'My debit card keeps getting declined', 'I cannot use my bank card in a store', 'Help me unblock my credit card', 'How do I activate a replacement debit card', 'Is there a way to activate a debit card online', 'My card never arrived by post', 'My wallet was stolen and my bank card is gone', 'I forgot the PIN for my payment card', 'My bank account has been frozen', 'How do I open a savings account', 'I want to close my checking account', 'The bank app crashes when I try to sign in', 'I forgot my mobile banking password', 'I need to reset my login details', 'My cash machine did not dispense banknotes', 'My salary transfer has not appeared', 'A wire transfer failed to arrive', 'I sent money to the wrong recipient', 'Can you help me cancel a bank transfer', 'I have a transaction I do not recognize', 'There is a suspicious charge on my statement', 'My account was charged twice', 'I cannot see a refund for a purchase', 'What should I do if an ATM keeps my card', 'My credit limit was reduced', 'How can I update my mailing address at the bank', 'I need an account statement', 'How do I get my credit card replaced', 'Why is my account balance lower than expected', 'There is missing money from my savings', 'Someone may have stolen funds from my account', 'A person knows my banking password', 'How can I report unauthorized withdrawals', 'I cannot make an international card payment', 'My account is under review and I cannot transfer', 'My payee has not received the deposit', 'How to enable online transactions on my card', 'My card was blocked after several attempts', 'How can I dispute a payment', 'How do I increase my ATM withdrawal limit', 'Can I get a refund of an accidental duplicate card charge', 'What do I do when my loan payment is missing', 'How can I request a bank statement for my mortgage', 'I sent the correct amount but the recipient got less', 'My direct debit has been charged unexpectedly', 'My UPI payment remains pending', 'Please tell me how to report card fraud', 'Can I pay my credit card bill early', 'My bank transfer was reversed', 'My card is not working', 'Someone took my money', 'Please tell me how to activate my new bank debit card']
NEGATIVE = ['Can the result of an election break my debit card', 'Can a football score change the balance in my account', 'Does the moon phase prevent my credit card from working', 'Will tomorrow being sunny cause my bank app to crash', 'Does washing my hands renew my debit card', 'Can eating ice cream make an ATM dispense more money', 'If I sing loudly will my account become unlocked', 'Does the color of my shirt decide whether my transfer goes through', 'Can a cricket player personally fix my password', 'Can a celebrity install my credit card by magic', 'Will taking a shower make the bank give me a loan', 'Could a history exam affect the PIN on my card', 'Will a famous actor cause my ATM withdrawal to fail', 'If I draw a picture can I increase my card limit', 'Can a dog barking cause my bank account to freeze', 'Does watching television change my card payment status', 'Would a thunderstorm change the number of digits in my PIN', 'Can my football team winning activate my card', 'Does a film award guarantee the bank refunds my money', 'If I brush my teeth will my bank transfer complete', 'Who won yesterday and why is my card called debit', 'Explain volcanoes and tell me what a debit card is', 'Write a short poem involving the word bank', 'Compose music about credit cards', 'Explain the physics of an ATM machine', 'Design a fictional debit card for my novel', 'What is the origin of the word bank', 'Translate the phrase bank card into German', 'Can you write Python code for a bank application', 'Make a joke about an ATM', 'Which movie stars use credit cards', 'Tell me a cricket score and mention a bank card', 'Does the ocean have banks and debit cards', 'What are the colors on a credit card design', 'Which famous athlete likes cash machines', 'Explain why a riverbank is eroding', 'How are bank buildings designed architecturally', 'Why does a video game character have a bank account', 'I want a drawing of a payment terminal', 'Which animals can recognize a credit card']
AMBIGUOUS = ['Can my friend install a debit card', 'Will a neighbor make my debit card work', 'Can my cousin put my debit card into the bank', 'Is it possible for my brother to set up a card for me', 'Does an athlete know how to enable a credit card', 'Could a teacher activate my banking account', 'Will a celebrity be able to arrange my ATM card', 'Can a stranger turn on my bank card', 'Can I load a debit card into a phone', 'How do you install a card', 'Is my card okay', 'Something weird is happening to my account', 'This bank thing seems broken', 'I have a problem with some money', 'I need to know about a card', 'Can my father fix this card for me', 'Do I need help to set up banking on a new device', 'Can a person other than me activate my replacement card', 'Why is money different today', 'Can someone make my transfer happen', 'Does my phone need to be installed for bank access', 'Can a bank card be installed physically', 'What if a person gives me a debit card', 'Could a friend perform card activation without my permission', 'My payment may or may not have worked', 'Something changed in my transaction', 'Who should I ask about my card', 'Is my bank behaving normally', 'Could a third party enable the banking app on my phone']

@lru_cache(maxsize=1)
def _model():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    x = POSITIVE + NEGATIVE + AMBIGUOUS
    y = ['banking']*len(POSITIVE) + ['unrelated']*len(NEGATIVE) + ['clarify']*len(AMBIGUOUS)
    pipe = Pipeline([('features', FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True))
    ])),('classifier',LogisticRegression(max_iter=1500,class_weight='balanced',C=2.5))])
    pipe.fit(x,y)
    return pipe

# Scores intentionally used as comparative signals, not probabilities of correctness.
def assess_request(message):
    q = message.strip()
    if not q:
        return {'kind':'unrelated','reason':'Empty message','scores':{}}
    pipeline = _model()
    distribution = dict(zip(pipeline.classes_,map(float,pipeline.predict_proba([q])[0])))
    # The positive decision requires an ample margin. Remaining cases fail closed,
    # preferably to a useful clarification, rather than choosing an incident.
    bank = distribution['banking']
    unrelated = distribution['unrelated']
    clarify = distribution['clarify']
    if bank >= 0.56 and bank-unrelated >= 0.18 and bank-clarify >= 0.12:
        kind = 'banking'
    elif unrelated >= 0.59 and unrelated-bank >= 0.19:
        kind = 'unrelated'
    else:
        kind = 'clarify'
    return {'kind':kind,'reason':'Trained request-purpose gate; uncertain requests are clarified',
            'scores':{k:round(v,3) for k,v in distribution.items()}}


def purpose_response(kind):
    if kind == 'unrelated':
        return ('ResolveAI handles banking customer-support requests only. '
                'Please describe a specific issue with a banking product or transaction. '
                'No account action was performed.')
    if kind == 'clarify':
        return ('Could you clarify the banking issue you want help with? '
                'For example, are you asking how to activate or use a card, '
                'or did an actual payment, login or transaction fail? '
                'I cannot infer an account or payment problem from the information provided. '
                'Please do not share any passwords, PINs or OTPs. '
                'No account action was performed. Human review is required.')
    raise ValueError(kind)

# Mixed-topic splitter identifies *two independently requested actions*;
# purchasing sports tickets with a card is only one banking question.
QUESTION_HEAD = re.compile(r'^(?:who|what|where|when|why|how|which|can|could|will|would|should|did|do|does|is|are|tell|explain|write|compose|create|recommend|help|check|summarize|calculate|find)\b', re.I)
BANK_TERMS = re.compile(r'\b(?:bank|banking|card|atm|account|balance|payment|transaction|refund|transfer|withdrawal|debit|credit|pin|otp|passcode|password|cash)\b', re.I)


def mixed_independent_requests(text):
    """Identify independent bank and nonbank *requests*, not incidental background."""
    # Prefer conservative splitting. Supports explicit conjunctions and punctuation.
    parts = re.split(r'\s+(?:and|also)\s+|[;?]\s*', text.strip(), flags=re.I)
    if len(parts) < 2:
        return False
    bank_request = nonbank_request = False
    for idx, clause in enumerate(parts):
        piece = clause.strip(' ,.!?')
        if not piece:
            continue
        has_request = bool(QUESTION_HEAD.match(piece))
        # A second coordinate clause commonly omits the question head:
        # "Why did X happen and how can I fix my card" still has both.
        if not has_request:
            continue
        if BANK_TERMS.search(piece):
            bank_request = True
        else:
            nonbank_request = True
    return bank_request and nonbank_request


def routing_veto(message, existing_domain=True):
    """Only veto automatically selected banking intents on strong counterevidence.

    Returns 'mixed_domain', 'purpose_uncertain', or None. An OOD domain
    rejection and high-risk safety routing must be handled in the caller.
    """
    if mixed_independent_requests(message):
        return 'mixed_domain'
    if not existing_domain:
        return None
    s = assess_request(message)['scores']
    if s['clarify'] >= 0.55 and s['banking'] < 0.22:
        return 'purpose_uncertain'
    if s['unrelated'] >= 0.57 and s['banking'] < 0.22:
        return 'purpose_uncertain'
    return None
