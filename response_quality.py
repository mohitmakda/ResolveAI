"""Human-readable, policy-scoped deterministic drafts for the fictional demonstration.

The wording below is NOT independently authoritative bank policy. A response is emitted
only after the calling pipeline approves the same category and cited policy.
"""

# Each entry contains only steps already present in the named fictional policy.
POLICY_DRAFTS = {
    'POL-01': 'If your payment card is lost or stolen, contact your bank immediately using a verified official channel and ask about blocking the card. Only the bank can confirm whether the card has been blocked.',
    'POL-02': 'If an ATM did not dispense the expected cash, keep the transaction details you have and report the issue through your bank’s verified support channel. The bank will need to investigate; this demonstration cannot confirm a refund or its status.',
    'POL-03': 'If you do not recognize a card payment, contact your bank’s verified fraud support promptly and ask about its official dispute process. Do not share authentication credentials.',
    'POL-04': 'If you are having trouble signing in, use your bank’s official account recovery process or contact verified support. If you suspect unauthorized access, contact the bank’s security support immediately instead of relying on ordinary recovery steps.',
    'POL-05': 'For a missing or delayed transfer, review the transaction details available through your bank’s official channel and contact verified support to request an investigation. The bank must confirm whether any reversal or refund has occurred.',
    'POL-07': 'If your card has not arrived, check any delivery information shown in your bank’s official app or verified channels. Contact bank support if the delivery status remains unclear; only the bank can confirm dispatch or arrival.',
    'POL-08': 'For a declined debit or credit card payment, first check your bank’s official app or verified notifications for any information about the failure. If the cause is unclear or the payment keeps failing, contact the bank through a verified support channel. Only the bank can confirm the reason or whether a restriction applies.',
    'POL-09': 'If you may have shared a password, PIN, CVV or one-time code, contact verified bank security support immediately and ask which protective or credential-replacement steps are appropriate. Do not share any credentials in this demonstration.',
    'POL-10': 'For a refund or reversal question, keep any available transaction information and contact your bank through a verified official channel to ask about its review process. The bank must confirm eligibility and current status; this demonstration cannot check your refund.',
    'POL-11': 'If your account is restricted, contact your bank through a verified channel to ask which identity checks or documents, if any, are required. The bank must confirm the reason for the restriction and whether it can be removed.',
    'POL-12': 'To update your contact or profile details, use your bank’s authenticated app or verified support channel and ask about the required verification process. This demonstration cannot change or confirm personal records.',
    'POL-13': 'For an expired, damaged, or replacement card, contact your bank through a verified channel to ask about its replacement process. The bank must confirm any fees, eligibility and delivery or activation status.',
    'POL-14': 'If a card transaction appears more than once or is unfamiliar, review the transaction information available through your bank’s official channels and request a review from verified support. The bank must confirm any dispute or reversal outcome.',
    'POL-15': 'If money has been taken from your account without permission, promptly contact your bank’s verified fraud or security team to report it and request an investigation. Ask the bank about protective steps. This demonstration cannot confirm recovery, reversal, or account security.',
}


def customer_draft(question, docs, category):
    """Render an approved one-policy customer draft or decline to invent missing details."""
    if not docs or not category:
        return ('I do not have enough approved information to advise on this request. '
                'Please ask an authorized bank support representative to review it. '
                'No account action was performed.')
    policy = docs[0]
    if policy.get('category') != category:
        return ('The available policy does not clearly match the issue. '
                'Please refer this request for human review. No account action was performed.')
    policy_id = policy.get('id')
    text = POLICY_DRAFTS.get(policy_id)
    if not text:
        return ('This request needs guidance from your bank’s verified support team. '
                'I cannot determine the correct next steps from the approved demonstration policy. '
                f'No account action was performed. Reference: [{policy_id}].')
    return f'{text} No account action has been performed here. Reference: [{policy_id}].'
