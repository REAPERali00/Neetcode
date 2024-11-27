import unittest
from main import Solution


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


if __name__ == "__main__":
    unittest.main()
