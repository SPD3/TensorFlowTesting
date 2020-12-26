import chess
import random

board = chess.Board()

print(board)
print(board.legal_moves)


def getRandomMove():
    numberOfLegalMoves = board.legal_moves.count()
    randomMoveIndex = random.randint(1, numberOfLegalMoves) - 1
    listOfMoves = list(board.legal_moves)
    return listOfMoves[randomMoveIndex]


def playRandomGame():
    while not board.is_game_over():
        board.push(getRandomMove())
        
    print(board)


playRandomGame()
print("result: ", board.result())
