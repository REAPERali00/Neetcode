from typing import List


class Solution:
    # checking if an array contains a duplicate
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False


sol = Solution()
print(sol.hasDuplicate([1, 2, 3, 4]))
