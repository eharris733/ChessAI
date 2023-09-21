import math
import random
import traceback
import chess.polyglot  # opening book
import chess
import chess.svg
from flask import Flask, Response, request
import webbrowser
import time



global movesSearched  # for debugging
global alreadyEvaluated  # transpositionTable
global currentStage #to keep track of which stage we are at in the game
alreadyEvaluated = {} #initializing our transposition table, which is just a python dictionary

#These are our piece tables
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, -10, -10, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]

#A homemade piece table that is only useful for endgames
kingsEndGameTable = [
    150, 100, 100, 100, 100, 100, 100, 150,
    100, 50, 50, 50, 50, 50, 50, 100,
    100, 50, 25, 25, 25, 25, 50, 100,
    100, 50, 25, 0, 0, 25, 50, 100,
    100, 50, 25, 0, 0, 25, 50, 100,
    100, 50, 25, 25, 25, 25, 50, 100,
    100, 50, 50, 50, 50, 50, 50, 100,
    150, 100, 100, 100, 100, 100, 100, 150,
]

#This function is very long, and was very hard to write.
# it returns the number of isolated and passed pawns for
#each side. I haven't found anything awry with it in testing
def pawnStructure(isLateEndgame):
    whiteIsolatedPawns = 0
    whitePassedPawns = 0

    blackPassedPawns = 0
    blackIsolatedPawns = 0

    listOfWhitePawns = {}
    listOfBlackPawns = {}
    # looking for which type of pawn, each pawn is
    # whiteScore
    for i in board.pieces(chess.PAWN, chess.WHITE):
        file = i % 8
        if (listOfWhitePawns.get(file) is not None):
            listOfWhitePawns[file] = listOfWhitePawns[file].append(math.ceil(i / 8))
        else:
            listOfWhitePawns[file] = [(math.ceil(i / 8))]
    for i in board.pieces(chess.PAWN, chess.BLACK):
        file = i % 8
        if (listOfBlackPawns.get(file) is not None):
            listOfBlackPawns[file] = listOfBlackPawns[file].append(math.ceil(i / 8))
        else:
            listOfBlackPawns[file] = [(math.ceil(i / 8))]

    for k in listOfWhitePawns:  # finds all the passed pawns for white
        whitePassedPawn = True
        if (listOfWhitePawns.get(k) is not None):
            if (k - 1 not in listOfWhitePawns and k + 1 not in listOfWhitePawns):  # check for isolated pawns
                whiteIsolatedPawns += len(listOfWhitePawns[k])

        if (k in listOfBlackPawns):  # This means there is another pawn on the file
            if (listOfWhitePawns.get(k) is not None and listOfBlackPawns.get(k) is not None):
                if (max(listOfWhitePawns[k]) < min(listOfBlackPawns[k])):  # not passed for white
                    whitePassedPawn = False

        else:
            for bk in listOfBlackPawns:  # for every black pawn
                inFront = False
                if abs(bk - k) == 1:  # if on adjacent file
                    if (listOfBlackPawns.get(bk) is not None and listOfWhitePawns.get(k) is not None):
                        if min(listOfBlackPawns[bk]) > max(listOfWhitePawns[
                                                               k]):  # if the black pawn is on a row behind the white, white pawn is passed
                            inFront = True

                if (inFront):
                    whitePassedPawn = False

        if (whitePassedPawn):
            whitePassedPawns += 1

    for k in listOfBlackPawns:  # finds all the passed pawns for white
        blackPassedPawn = True
        if (listOfBlackPawns.get(k) is not None):
            if k - 1 not in listOfBlackPawns and k + 1 not in listOfBlackPawns:  # check for isolated pawns
                blackIsolatedPawns += len(listOfBlackPawns[k])

        if (k in listOfWhitePawns):  # This means there is another pawn on the file
            if (listOfWhitePawns.get(k) is not None and listOfBlackPawns.get(k) is not None):
                if (min(listOfBlackPawns[k]) >= max(listOfWhitePawns[k])):  # not passed for white
                    blackPassedPawn = False

        else:
            for wk in listOfWhitePawns:  # for every black pawn
                inFront = False
                if abs(wk - k) == 1:  # if on adjacent file
                    if (listOfWhitePawns.get(wk) is not None and listOfBlackPawns.get(k) is not None):
                        if min(listOfWhitePawns[wk]) > max(listOfBlackPawns[
                                                               k]):  # if the black pawn is on a row behind the white, white pawn is passed
                            inFront = True

                if (inFront):
                    blackPassedPawn = False

        if (blackPassedPawn):
            blackPassedPawns += 1

    pawnScore = 100 * (whitePassedPawns - blackPassedPawns) + -10 * (whiteIsolatedPawns - blackIsolatedPawns)
    if(isLateEndgame):
        pawnScore *= 5 #we care a lot more about pawns in the endgame
    return pawnScore

#The main evaluation function
def evaluationFunction():
    if board.is_checkmate():  # checkmate is the most important thing, and we need to check for draws first
        if board.turn:
            return -9999999
        else:
            return 9999999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    if board.is_game_over():
        return 0

    wp = len(board.pieces(chess.PAWN, chess.WHITE)) #number of each type of piece for each side
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))


    material = 100 * (wp - bp) + 310 * (wn - bn) + 330 * (wb - bb) + 450 * (wr - br) + 900 * (wq - bq) #attach material weights to the differences in material

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])
    kingSQWhite = sum([kingsEndGameTable[i] for i in board.pieces(chess.KING, chess.WHITE)])

    kingSQBlack = sum([kingsEndGameTable[i] for i in board.pieces(chess.KING, chess.BLACK)])#don't need to mirror it cause its symmetrical

    # we create a king safety score and a pawn stucture score.
    # in the middlegame king safety score is much more important than in the endgame
    # in the endgame, pawn structure score ismuch more important than in the middlegame
    # using these overgeneralizations, we can better evaluate moves.
    # lastly, if we are in the corner king stage, all we care about is cornering the king and taking away moves.
    # This is important as checkmate may be 20 moves away, but we have a rook and they have nothing, so we need to find mate
    global currentStage
    if (currentStage == 'middlegame'):
        eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq + pawnStructure(False) + kingSafetyWhite() - kingSafetyBlack()
    elif (
            currentStage == 'endgame'):  # undervalue king safety, and overvalue pawn structure, undervalue piece placement a bit
        eval = material + .8 * (pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq) + 2 * pawnStructure(False) + .5 * (
                     kingSafetyWhite() - kingSafetyBlack()
        )  # we care more about pawnstructure
    elif(currentStage == 'cornerkingwhite'):  # cornerKingTime, disregard piece placement, we want to hunt the king without losing our pieces
        eval = material + kingSQWhite - getDistance(board.king(chess.WHITE), board.king(chess.BLACK)) + pawnStructure(True)

    else:
        eval = material + kingSQBlack + getDistance(board.king(chess.WHITE), board.king(chess.BLACK)) + pawnStructure(True)


    if board.turn:
        return eval #for white
    else:
        return -eval #for black


# manhattan distance
def getDistance(index1, index2):
    file1 = index1 % 7
    file2 = index2 % 7
    rank1 = index1 >> 3
    rank2 = index2 >> 3
    rankDistance = abs(rank2 - rank1)
    fileDistance = abs(file2 - file1)
    return rankDistance + fileDistance

#Higher the better for king safety. Low numbers is equal to very safe
def kingSafetyBlack():
    kingTropismValue = 0
    for i in board.pieces(chess.PAWN, chess.WHITE):
        kingTropismValue += 2 * getDistance(i, board.king(chess.BLACK))
    for i in board.pieces(chess.KNIGHT, chess.WHITE):
        kingTropismValue += 2 * getDistance(i, board.king(chess.BLACK))
    for i in board.pieces(chess.BISHOP, chess.WHITE):
        kingTropismValue += 2 * getDistance(i, board.king(chess.BLACK))
    for i in board.pieces(chess.ROOK, chess.WHITE):
        kingTropismValue += 3 * getDistance(i, board.king(chess.BLACK))
    for i in board.pieces(chess.QUEEN, chess.WHITE):
        kingTropismValue += 5 * getDistance(i, board.king(chess.BLACK))
    return kingTropismValue

#This function mirrors the above just for white instead
def kingSafetyWhite():
    kingTropismValue = 0
    for i in board.pieces(chess.PAWN, chess.BLACK):
        kingTropismValue += getDistance(i, board.king(chess.WHITE))
    for i in board.pieces(chess.KNIGHT, chess.BLACK):
        kingTropismValue += 2 * getDistance(i, board.king(chess.WHITE))
    for i in board.pieces(chess.BISHOP, chess.BLACK):
        kingTropismValue += 2 * getDistance(i, board.king(chess.WHITE))
    for i in board.pieces(chess.ROOK, chess.BLACK):
        kingTropismValue += 3 * getDistance(i, board.king(chess.WHITE))
    for i in board.pieces(chess.QUEEN, chess.BLACK):
        kingTropismValue += 5 * getDistance(i, board.king(chess.WHITE))
    return kingTropismValue


# This is a homemade function that tries to detect whether we are in the opening, middle, or endgame.
# Important because we want to evaluate positions very differently for all three
# I stole some of the values from lichess engine
def detectStageOfGame():
    # endgame will start if there are a total of 5 or less non-pawn/non-kings for my purposes,
    # or a total of 6 or less non-pawn/non-kings and no queens
    minorPieceSum = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(
        board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + len(
        board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
    queenSum = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.WHITE))
    if (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.WHITE))) == 0:
        return 'cornerkingwhite'
    elif(len(board.pieces(chess.QUEEN, chess.BLACK)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + len(board.pieces(chess.ROOK, chess.BLACK))) == 0:
        return 'cornerkingblack'
    elif (minorPieceSum +
        queenSum) <= 3 or (minorPieceSum <= 4):
        return 'endgame'
    return 'middlegame' # note that we don't care abt opening because we are using an opening book for most curcial opening moves





# negamax from internet, a recursive way to implement min max https://www.chessprogramming.org/Negamax
def searchMinMax(depth):
    if depth == 0:
        return evaluationFunction()
    max = -99999
    for i in board.generate_legal_moves():
        board.push(i)
        score = -1 * searchMinMax(depth - 1)
        board.pop()
        if (score > max):
            max = score
    return max


# The goal of this function is to sort our list by what is most likely to be played, very rudimentary sorting algorithim
def captureFirstHueristic(listOfMoves):
    tempList = []
    for i in listOfMoves:
        if ((i in board.generate_legal_captures() or i in board.generate_castling_moves())):
            tempList.insert(0, i)
        else:
            tempList.append(i)
    return tempList

#homemade heuristic to order our moves so we don't start looking down bad trees first
def pvAndCaptureHueristic(listOfMoves, pvMove):
    tempList = []
    for i in listOfMoves:
        if ((i in board.generate_legal_captures() or i in board.generate_castling_moves())):
            tempList.insert(0, i)
        else:
            tempList.append(i)
    if (pvMove != None):
        tempList.insert(0, pvMove)  # we insert the principal variation move at the beggining
    return tempList


# this is the negamax implementation of alpha beta pruning.
#I borrowed a lot of ideas from chessprogamming.com
def alphaBeta(depth, alpha, beta):
    global movesSearched, alreadyEvaluated
    bestScore =-99999
    if (depth == 0):
        return quietConfirmation(alpha, beta)  # quietConfirmation(alpha, beta)

    for i in captureFirstHueristic(board.generate_legal_moves()):  # sorting the moves first might save us some time
        board.push(i)
        movesSearched += 1
        if (board.board_fen() + str(i) + str(
                board.turn)) in alreadyEvaluated:  # our key is the board position, plus the move we make
            score = alreadyEvaluated.get(board.board_fen() + str(i) + str(board.turn))
        else:
            score = -alphaBeta(depth - 1, -beta, -alpha, )
            # try and guess that all moves will be better
        board.pop()

        if (score >= beta):
            return score
        if (score > bestScore):
            bestScore = score
            if (score > alpha):
                alpha = score

    return bestScore


# I used https://www.chessprogramming.org/Quiescence_Search as a reference for this function
# this is the function we use at end of a search to make sure we are only evaluating
# quiet position. If we were to be evaluation positions where a capture or check is possible,
# we could be stopping one node too short and get disastrous results

def quietConfirmation(alpha, beta):
    global movesSearched
    baseline = evaluationFunction()
    if (baseline >= beta):  # This code is very similar to alpha beta
        return beta
    if (alpha < baseline):
        alpha = baseline
    for i in board.generate_legal_captures():  # this cycles through every capture
        board.push(i)
        movesSearched += 1
        tempScore = -quietConfirmation(-beta, -alpha)
        board.pop()
        if (tempScore >= beta):
            return beta
        if (tempScore > alpha):
            alpha = tempScore
    return alpha


# This is without pruning, for comparison and testing
def selectMoveMinMax(depth):
    global movesSearched
    bestMove = None
    bestScore = -99999
    for i in board.generate_legal_moves():
        board.push(i)
        movesSearched += 1  # for debugging
        score = -searchMinMax(depth)  # this our minmax algorithim
        if (score > bestScore):
            print(score)
            bestScore = score
            bestMove = i
        board.pop()
    return bestMove

#For testing purposes
def selectRandomMove():
    return random.choice(list(board.generate_legal_moves()))


#Function to use opening book should I so choose
def useOpeningBook():
    global exhaustedOpeningTable
    try:
        move = chess.polyglot.MemoryMappedReader("human.bin").weighted_choice(
            board).move
        return move
    except IndexError:
        print('could not find a move')
        exhaustedOpeningTable = True
        return False

#Bread and butter function. Uses iterative deepening calls to our alphabeta search
def iterativeDeepeningAlphaBeta(timeLimit):
    global movesSearched, currentStage
    currentStage = detectStageOfGame() #we only want to detect the stage of the game once, not every single time we evaluate a position
    movesSearched = 0
    depth = 2 #We initially search at depth 2 instead of 1 to save time
    bestMove = selectMoveAlphaBeta(depth, None)
    lastBestMove = bestMove
    initTime = time.time()
    while (time.time() - initTime) < timeLimit and exhaustedOpeningTable:
        depth += 1
        bestMove = selectMoveAlphaBeta(depth, bestMove)
        if (bestMove == None):
            bestMove = lastBestMove
            break  # This means we've gone so deep there are no moves to choose from
        else:
            lastBestMove = bestMove
    print(
        'last move ' + str(bestMove) + ' looked through ' + str(movesSearched) + ' moves ')
    return bestMove


# Takes in a depth and principal variation Move, which is the move that is assumed to be best found at the previous depth.
def selectMoveAlphaBeta(depth, pvMove):
    global movesSearched, alreadyEvaluated
    if (not exhaustedOpeningTable):  # once we fail once, we stop checking for opening moves to save time
        tempMove = useOpeningBook()
        if (tempMove != False):
            return tempMove
    bestScore = -99999
    bestMove = None
    alpha = -100000
    beta = 100000
    for i in pvAndCaptureHueristic(board.generate_legal_moves(), pvMove):
        board.push(i)
        movesSearched += 1
        if (board.board_fen() + str(i) + str(
                board.turn) + str(depth)) in alreadyEvaluated:  # our key is the board position, plus the move we make
            score = alreadyEvaluated.get(board.board_fen() + str(i) + str(board.turn) + str(depth))
        else:
            score = -alphaBeta(depth - 1, -beta, -alpha)
        if (score > bestScore):
            bestScore = score
            bestMove = i
        if (alpha < score):
            alpha = score
        alreadyEvaluated[board.board_fen() + str(i) + str(board.turn) + str(depth)] = bestScore
        board.pop()
    print(str(bestMove) + ' found at depth ' + str(depth))
    return bestMove

app = Flask(__name__)


# Front Page of the Flask Web Page
@app.route("/")
def main():
    global  board, exhaustedOpeningTable
    ret = '<html><head>'
    ret += '<style>input {font-size: 20px; } button { font-size: 20px; }</style>'
    ret += '</head><body>'
    ret += '<img width=510 height=510 src="/board.svg?%f"></img></br></br>' % time.time()
    ret += '<form action="/game/" method="post"><button name="New Game" type="submit">New Game</button></form>'
    ret += '<form action="/undo/" method="post"><button name="Undo" type="submit">Undo Last Move</button></form>'
    ret += '<form action="/move/"><input type="submit" value="Make Human Move:"><input name="move" type="text"></input></form>'
    ret += '<form action="/dev/" method="post"><button name="Comp Move" type="submit">Make Ai Move</button></form>'
    ret += '<form action="/rand/" method="post"><button name="Comp Move" type="submit">Make Random Move</button></form>'
    return ret


# Display Board
@app.route("/board.svg/")
def board():
    return Response(chess.svg.board(board=board, size=700), mimetype='image/svg+xml')


# Human Input Move
@app.route("/move/")
def move():
    try:
        move = request.args.get('move', default="")
        board.push_san(move)
    except Exception:
        traceback.print_exc()
    return main()


# Make Ai’s Move
@app.route("/dev/", methods=['POST'])
def dev():
    try:
        global currentStage
        board.push(iterativeDeepeningAlphaBeta(3)) #iterative deepening for 3 seconds, then finish the depth that it reaches
        print(currentStage)
        print(evaluationFunction())
    except Exception:
        traceback.print_exc()
    return main()


# Make Random Move
@app.route("/rand/", methods=['POST'])
def rand():
    try:
        board.push(selectRandomMove())

    except Exception:
        traceback.print_exc()
    return main()


# New Game
@app.route("/game/", methods=['POST'])
def game():
    global exhaustedOpeningTable
    board.reset()
    exhaustedOpeningTable = False  # reset this
    return main()


# Undo
@app.route("/undo/", methods=['POST'])
def undo():
    try:
        board.pop()
    except Exception:
        traceback.print_exc()
    return main()


#runs the app
if __name__ == '__main__':
    board = chess.Board() #create an empty board

    exhaustedOpeningTable = False
    movesSearched = 0

    webbrowser.open("http://127.0.0.1:5000/")
    app.run()
