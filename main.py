from typing import List
from collections import Counter


class Solution:
    # 1. checking if an array contains a duplicate
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        adds = {}
        for i, num in enumerate(nums):
            if num in adds:
                return [adds[num], i]
            adds[target - num] = i


sol = Solution()
print(sol.isAnagram("racecar", "carrace"))
