from typing import List
from collections import Counter, defaultdict


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
        return []

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anags = defaultdict(list)
        for s in strs:
            key = "".join(sorted(s))
            anags[key].append(s)
        return list(anags.values())

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return [x for x, count in Counter(nums).most_common(k)]


sol = Solution()
print(sol.topKFrequent([1, 2, 2, 3, 3, 3], 2))
