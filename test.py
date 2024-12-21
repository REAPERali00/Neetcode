import unittest
from main import Solution, ListNode


class TestLeet(unittest.TestCase):

    def setUp(self):
        self.solution = Solution()

    def testIsValidSudoku(self):
        test_cases = [
            (
                [
                    ["1", "2", ".", ".", "3", ".", ".", ".", "."],
                    ["4", ".", ".", "5", ".", ".", ".", ".", "."],
                    [".", "9", "8", ".", ".", ".", ".", ".", "3"],
                    ["5", ".", ".", ".", "6", ".", ".", ".", "4"],
                    [".", ".", ".", "8", ".", "3", ".", ".", "5"],
                    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
                    [".", ".", ".", ".", ".", ".", "2", ".", "."],
                    [".", ".", ".", "4", "1", "9", ".", ".", "8"],
                    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
                ],
                True,  # Expected result
            ),
            (
                [
                    ["1", "2", ".", ".", "3", ".", ".", ".", "."],
                    ["4", ".", ".", "5", ".", ".", ".", ".", "."],
                    [".", "9", "1", ".", ".", ".", ".", ".", "3"],
                    ["5", ".", ".", ".", "6", ".", ".", ".", "4"],
                    [".", ".", ".", "8", ".", "3", ".", ".", "5"],
                    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
                    [".", ".", ".", ".", ".", ".", "2", ".", "."],
                    [".", ".", ".", "4", "1", "9", ".", ".", "8"],
                    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
                ],
                False,  # Expected result
            ),
        ]

        for i, (board, expected) in enumerate(test_cases):
            with self.subTest(f"Test case {i+1}: "):
                self.assertEqual(self.solution.isValidSudoku(board), expected)

    def testMergeTwoLists(self):
        def list_to_linked_list(lst):
            """Convert list to linked list."""
            dummy = ListNode()
            current = dummy
            for val in lst:
                current.next = ListNode(val)
                current = current.next
            return dummy.next

        def linked_list_to_list(node):
            """Convert linked list to list."""
            result = []
            while node:
                result.append(node.val)
                node = node.next
            return result

        # Test cases
        test_cases = [
            ([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]),  # Expected result
        ]

        for i, (list1, list2, expected) in enumerate(test_cases):
            with self.subTest(f"Test case {i+1}:"):
                l1 = list_to_linked_list(list1)
                l2 = list_to_linked_list(list2)
                result = self.solution.mergeTwoLists(l1, l2)
                self.assertEqual(linked_list_to_list(result), expected)


if __name__ == "__main__":
    unittest.main()
