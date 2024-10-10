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

### [Top K Elements in List](https://neetcode.io/problems/top-k-elements-in-list)

Problem: Given an integer array nums and an integer k, return the k most frequent elements within the array.

The test cases are generated such that the answer is always unique.

You may return the output in any order.

Solution: we start by counting the frequency of numbers in the array, and then sorting them to find the last k elements that are repeated. Fortunately, we have an easy way to do this in python using Counters. Counter not only gives us the count for each number, but it also has a function called most-common which would return the n most repeated numbers, in (num, count) format

```python
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return [x for x, count in Counter(nums).most_common(k)]
```

or we can use dictionary:

```python

from typing import List

def topKFrequent(nums: List[int], k: int) -> List[int]:
    # Step 1: Create a frequency dictionary
    freq = {}
    for num in nums:
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1

    # Step 2: Sort the dictionary by values in descending order
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Extract the top k elements (keys only)
    return [num for num, count in sorted_freq[:k]]

```

### [String Encode and Decode](https://neetcode.io/problems/string-encode-and-decode)

Problem: Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.
Please implement encode and decode

Solution: we first encode the string by adding the length of the string followed by a special character, : for example, and when decoding we will look for this special character and mark its location and grab the number representing its length. This way, we can find the exact length and location of the string and add it to the list!

```python
  def encode(self, strs: List[str]) -> str:
      return "".join(f"{len(s)}:{s}" for s in strs)

  def decode(self, s: str) -> List[str]:
      i = 0
      result = []
      while i < len(s):
          j = s.find(":", i)
          strLen = int(s[i:j])
          result.append(s[j + 1 : j + 1 + strLen])
          i = j + 1 + strLen
      return result
```

### [Products of array discluding self](https://neetcode.io/problems/products-of-array-discluding-self)

Problem:

Given an integer array nums, return an array output where output[i] is the product of all the elements of nums except nums[i].

Each product is guaranteed to fit in a 32-bit integer.

Follow-up: Could you solve it in O(n)O(n) time without using the division operation?

Solution:

We can solve this by splitting the array into two parts: the part before the number and the part after the number (_ie, the prefix and suffix_).

As we are going through the array we first do the multiplication of the suffix, adding the products before adding the product of the number, and then do the same for the suffix!

```python
  def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        output = [1] * n
        prefix = 1
        suffix = 1

        for i in range(n):
            output[i] *= prefix
            prefix *= nums[i]

        for i in range(n - 1, -1, -1):
            output[i] *= suffix
            suffix *= nums[i]

        return output
```
