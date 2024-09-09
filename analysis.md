# [Neetcode 150](https://neetcode.io/practice)

## Arrays and hashing

### Contains duplicate

problem: Given an integer array nums, return true if any value appears more than once in the array, otherwise return false.

solution => either use a sort and then check the next value (nlogn) or use a set to check for unique values (n)

```python
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

```

### Is Anagram

P: Given two strings s and t, return true if the two strings are anagrams of each other, otherwise return false.

An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

S: one way is to create a dictionary, loop over one set of strings and get the counts. then loop over the other string and deduct the count for each character found, if by the end the count for each character is not 0 then its not an anagram.

```python
def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    charCount = {}

    for sval in s:
        if sval in charCount:
            charCount[sval] += 1
        else:
            charCount[sval] = 1

    for tval in t:
        if tval in charCount:
            charCount[tval] -= 1
        else:
            return False

    for count in charCount.values():
        if count != 0:
            return False

    return True

```

the other way, is to seemply use counter from collections:

```python
from collections import Counter

def isAnagram(self, s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

```

### Two Integer Sum

P: Given an array of integers nums and an integer target, return the indices i and j such that nums[i] + nums[j] == target and i != j.

You may assume that every input has exactly one pair of indices i and j that satisfy the condition.

Return the answer with the smaller index first.
S: create a dictionary of the number needed to complement at the index, if the number is found in the dict then we have the two indexes we need

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
        adds = {}
        for i, num in enumerate(nums):
            if num in adds:
                return [adds[num], i]
            adds[target - num] = i
```

### Anagram Groups

P: Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.

An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

S: we start by creating a default dict, a dictinary that its keys point to a list. we will go through each string, and sort them (the join is to change the results of sort from list to string) to get the anagram value for all strings. if its not seen before, appending to dict will create a new list. if it is seen before, then it will add it to the key(which all anagrams sorted will be the same) list. that is simply it!

```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anags = defaultdict(list)
        for s in strs:
            key = "".join(sorted(s))
            anags[key].append(s)
        return list(anags.values())
```
