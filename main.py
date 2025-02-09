import collections
from typing import List, Optional
from collections import Counter, defaultdict, deque
import heapq
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, right=None, left=None):
        self.val = val
        self.right = right
        self.left = left


class Solution:
    # 1. checking if an array contains a duplicate
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

    # check letter count, both must be same 
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

    # is there two numbers in this array, where x+y=target
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        adds = {}
        for i, num in enumerate(nums):
            if num in adds:
                return [adds[num], i]
            adds[target - num] = i
        return []

    # group words that have the same letter count
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anags = defaultdict(list)
        for s in strs:
            key = "".join(sorted(s))
            anags[key].append(s)
        return list(anags.values())

    # return a  list of the top k most repeated numbers in the list  
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return [x for x, count in Counter(nums).most_common(k)]
    
    # Encode a list to a single string, and then decode it
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

    # create an array that is  composed of product of all elements but i 
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

    # check if a matrix is a valid suodku, meaning no repeat in row, col, and 3x3
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set)  # key: (r //3, c // 3)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (
                    board[r][c] in rows[r]
                    or board[r][c] in cols[c]
                    or board[r][c] in squares[(r // 3, c // 3)]
                ):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])
        return True

    # reverse a linked list
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, current = None, head
        while current:
            next = current.next
            current.next = prev
            prev = current
            current = next
        return prev

    # Merge to sorted linked lists to a sorted linked list
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dumby = node = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next

        node.next = list1 or list2

        return dumby.next

    # 
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head:
            return None

        # Find the middle point
        slow, fast = head, head.next
        while fast and fast.next:  # is fast necessary?
            slow = slow.next
            fast = fast.next.next
        second = slow.next
        prev = slow.next = None

        # reverse the second half
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
        first, second = head, prev

        # go by on from left, one from right and insert the new nodes
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        tmp = root.left
        root.left = root.right
        root.right = tmp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0  # made the variable global for functino access

        # Gives the height of the node
        def dfs(curr):
            if not curr:
                return 0
            left = dfs(curr.left)
            right = dfs(curr.right)
            self.diameter = max(
                self.diameter, left + right
            )  # record the largest height
            return 1 + max(
                left, right
            )  # the height of the parent node is the larger between left and right

        dfs(root)
        return self.diameter

    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def dfs(curr):
            if not curr:
                return [
                    True,
                    0,
                ]  # create a list as a return type, store both the current state and height
            left, right = dfs(curr.left), dfs(curr.right)
            balance = (
                left[0] and right[0] and abs(left[1] - right[1]) <= 1
            )  # check if left and right were prev balanced, and check for new balance
            return [balance, 1 + max(left[1], right[1])]

        return dfs(root)[0]

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not q or not p:
            return not (q or p)
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        def is_same(main, sub):
            if not main or not sub:
                return not (main or sub)
            if main.val != sub.val:
                return False
            return is_same(main.left, sub.left) and is_same(main.right, sub.right)

        if is_same(root, subRoot):
            return True
        if not root:
            return False
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        # the moment a split occurs, the node becomes a common Ancestor
        if not root or not p or not q:
            return None
        if max(q.val, p.val) < root.val:
            return self.lowestCommonAncestor(root.left, p, q)

        elif min(q.val, p.val) > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        que = deque([root])
        while que:
            qlen = len(que)
            level = []
            for _ in range(qlen):
                node = que.popleft()
                if node:
                    level.append(node.val)
                    que.append(node.left)
                    que.append(node.right)
            if level:
                res.append(level)
        return res

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        que = deque([root])
        while que:
            right = None
            qlen = len(que)
            for _ in range(qlen):
                node = que.popleft()
                if node:
                    right = node
                    que.append(node.left)
                    que.append(node.right)
            if right:
                res.append(right.val)
        return res

    def goodNodes(self, root: TreeNode) -> int:
        def dfs(curr, max):
            if not curr:
                return 0

            return (
                (1 if curr.val >= max else 0)
                + dfs(curr.left, max(max, curr.val))
                + dfs(curr.right, max(max, curr.val))
            )

        return dfs(root, root.val)

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head: 
            return False
        slow, fast = head, head
        while fast and fast.next: 
            slow = slow.next 
            fast = fast.next.next
            if slow == fast:
                return True 
        return False 
         
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dumby = ListNode()
        dumby.next = head
        found, ahead = dumby, dumby 
        for _ in range(n+1): 
            ahead = ahead.next
        while ahead: 
            found = found.next 
            ahead = ahead.next 
        found.next = found.next.next 
        return dumby.next

    # TODO: This one is Brilliant 
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        originToCopy = {None:None}
        curr = head
        # map original to copy using hashmap 
        while curr: 
            # create a copy, but not assign links 
            copy = Node(curr.val)
            originToCopy[curr] = copy
            curr = curr.next 

        # because random pointer is in the original keys, we can use it to find the copies address! 
        curr = head
        while curr: 
            copy = originToCopy[curr]
            copy.next = originToCopy[curr.next]
            copy.random = originToCopy[curr.random]
            curr = curr.next
        return originToCopy[head]

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dumby = copy = ListNode()
        carry = 0
        while l1 or l2: 
            copy.next = ListNode()
            copy = copy.next
            l = l1.val if l1 else 0 
            r = l2.val if l2 else 0 
            copy.val, carry = ( (l + r+ carry)%10  , (l + r+ carry)//10 )

            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if carry: 
            copy.next = ListNode(carry)
            copy = copy.next
        return dumby.next

    # Warning : this is o(1) space, so we need to use Linked list + Floyd's algorithm 
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0 # we can always start at zero, since its not going to be part of the loop (nums start at 1)
        while True: 
            slow = nums[slow] # Elements are indexes of the array 
            fast = nums[nums[fast]] 
            if fast == slow: break  #Find the point where the fast and slow pointers meet
        slow2 = 0  # start another slow pointer
        while True: 
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2: return slow  #once the two meet, we have found the solution 

    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-x for x in stones]
        heapq.heapify(stones)
        while len(stones) > 1: 
            y = heapq.heappop(stones) * -1
            x = heapq.heappop(stones) * -1
            if x < y : 
                heapq.heappush(stones, x-y)

            elif x > y : 
                heapq.heappush(stones, y-x)
        return -stones[0] if len(stones) else 0 

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        def get_dist(point: List[int]): 
            x = point[0]
            y = point[1]
            return math.sqrt((x) ** 2 + (y) **2)
        distances = [(-get_dist(point), point) for point in points]
        heapq.heapify(distances)
        while len(distances) > k: 
            heapq.heappop(distances)
        return [y for _,y in distances]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        while len(nums) > k: 
            heapq.heappop(nums)
        return nums[0]

    # TODO: This is a great example of que, map and heap
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = Counter(tasks)
        maxHeap = [-cnt for cnt in count.values()]
        heapq.heapify(maxHeap)

        time = 0
        q = deque()  # pairs of [-cnt, idleTime]
        while maxHeap or q:
            time += 1

            if not maxHeap:
                time = q[0][1]
            else:
                cnt = 1 + heapq.heappop(maxHeap)
                if cnt:
                    q.append([cnt, time + n])
            if q and q[0][1] == time:
                heapq.heappush(maxHeap, q.popleft()[0])
        return time 

    # TODO a good graph intro
    def numIslands(self, grid: List[List[str]]) -> int:
        # find a 1, if found add it to the visited. then go through all nodes that are one, add to que till que empty.
        if not grid: 
            return 0 
        islands = 0
        rows, cols = len(grid), len(grid[0])
        visited = set() 

        # go through each 1 and add its surrounding ones till none is left
        def bfs(r,c): 
            q = deque()
            visited.add((r,c))
            q.append((r,c))
            directions = [[1,0], [-1,0], [0,1], [0,-1]]
            while q: 
                row, col = q.popleft()
                for dr, dc in directions: 
                    r,c = row + dr, col + dc
                    if (r in range(rows) and 
                        c in range(cols) and
                        grid[r][c] == "1" and
                        (r, c) not in visited):
                        q.append((r,c))
                        visited.add((r,c))

        for r in range(rows): 
            for c in range(cols): 
                if grid[r][c] == "1" and (r,c) not in visited: 
                    bfs(r,c)
                    islands += 1

        return islands
                    

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid: 
            return 0 
        rows , cols = len(grid), len(grid[0])
        visited = set()
        max_area = 0 

        def bfs (r, c): 
            q = deque()
            visited.add((r,c))
            q.append((r,c))
            dirs = [[1,0], [-1,0],[0,1],[0,-1]]
            count = 1
            while q: 
                row, col = q.popleft()
                for dr, dc in dirs: 
                    r, c = row + dr, col + dc
                    if (r in range(rows) and 
                        c in range(cols) and 
                        grid[r][c] == 1 and 
                            (r,c) not in visited): 
                        q.append((r,c))
                        visited.add((r,c))
                        count+=1
            return count

        for r in range(rows): 
            for c in range(cols): 
                if grid[r][c] == 1 and (r,c) not in visited: 
                    area = bfs(r,c) 
                    max_area = max(area, max_area) 


        return max_area
    
    
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        originToCopy = {}

        def dfs(node): 
            if node in originToCopy: 
                return originToCopy[node]
            cp = Node(node.val)
            originToCopy[node] = cp
            for n in node.neighbors: 
                cp.neighbors.append(dfs(n))
            return cp
        return dfs(node) if node else None

    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        if not grid: 
            return None

        rows, cols = len(grid), len(grid[0])
        visited = set()
        q = deque()

        # add all treasures to the que 
        for r in range(rows): 
            for c in range(cols): 
                if grid[r][c] == 0: 
                    q.append((r,c))
                    visited.add((r,c))

        dist = 0 
        while q: 
            for _ in range(len(q)): 
                row,col = q.popleft()
                grid[row][col] = dist

                for dr, dc in [[1,0],[-1,0],[0,1],[0,-1],]: 
                    nr, nc = row + dr, col + dc
                    if (nr not in range(rows) or 
                        nc not in range(cols) or 
                        grid[nr][nc] == -1 or 
                            (nr, nc) in visited): continue 
                    q.append((nr,nc))
                    visited.add((nr,nc))

            dist +=1

    # TODO lol this is Fibonacci 
    def climbStairs(self, n: int) -> int:
        one, two = 1,1 
        for i in range(n-1): 
            tmp = two 
            two = one + two 
            one = tmp
        return two
    
    def rob(self, nums: List[int]) -> int:
        # we are going to store max ammount in each step 
        rob1, rob2 = 0,0
        for n in nums: 
            temp = rob2
            rob2  = max(rob1+n, rob2) # see what is the max ammount we can rob at that index, 
            #which is compare which is bigger: current number or the addition of the surrounding numbers
            rob1 = temp #move the prev
        return rob2

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(len(cost)-3, -1, -1) : 
            cost[i] += min(cost[i+1], cost[i+2])
        return min(cost[0],cost[1])

    def rob2(self, nums: List[int]) -> int:
        return max(nums[0], self.rob(nums[1:]), 
                            self.rob(nums[:-1]))
    
    def longestPalindrome(self, s: str) -> str:
        res= ""

        for i in range(len(s)): 
            # odd case, ex: aba
            l = r = i
            while l >= 0 and r < len(s) and s[r] == s[l]: 
                res = s[l:r+1] if len(res) < r-l+1 else res
                r +=1 
                l -= 1
        
            # Even case, ex: abba
            l , r = i, i+1
            while l >= 0 and r < len(s) and s[r] == s[l]: 
                res = s[l:r+1] if len(res) < r-l+1 else res
                r +=1 
                l -= 1
        return res

    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0] 
        curr_sum = 0 
        for num in nums: 
            if curr_sum < 0: #if the addition  prior was negative, it doesn't help is so we reset and ignore them
                curr_sum = 0 
            curr_sum += num 
            max_sum = max(max_sum, curr_sum)
        return max_sum

    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) -1 
        for i in range(len(nums)-2, -1, -1): 
            if i + nums[i] >= goal: 
                goal = i  

        return goal == 0 

    def jump(self, nums: List[int]) -> int:
        res = l = r = 0 
        while r < len(nums)- 1: 
            farthest = 0 
            for i in range(l, r+1): 
                farthest = max(farthest, i + nums[i])
            l = r+1
            r = farthest 
            res +=1 

        return res
        
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost): return -1
        res = tot = 0 
        for i in range(len(gas)): 
            tot += gas[i]- cost[i]
            if tot < 0:
                tot =0 
                res = i+1
        return res



