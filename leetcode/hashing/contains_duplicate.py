# tags: hashing, set
# A past solved solution kept as a PATTERN REFERENCE / analogy.
# The system retrieves files like this to remind you of techniques you
# have used before — it does not copy them into answers.


def contains_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
