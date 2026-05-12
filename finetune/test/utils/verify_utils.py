"""Math verification utilities"""

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def math_verify_compare(answer, ground_truth):
    if answer.strip().lower() == ground_truth.strip().lower():
        return True
    try:
        return verify(parse(str(ground_truth)), parse(str(answer)))
    except:
        try:
            return verify(parse(ground_truth), parse(answer))
        except:
            return answer.strip().lower() == ground_truth.strip().lower()
