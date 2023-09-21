# ChessAI
 A somewhat Clever Chess AI that utilizes alpha-beta pruning, move order, and more to result in about a 1500 elo playing strength. 
## A game versus my friend 
![A gif of a game my engine played versus my friend](https://github.com/ChessAI/MyEngineInAction.gif)

Note that 
A. My friend is about 1400 elo on Chess.com
B. It stalemates despite a dominant position
 
## Technical Specifications
Written completely in Python, the engine has a completely running Flask app that allows one to play against the engine. There is also the option to include Stockfish, which allows one to see what a more efficient algorithm comes up with. 

 ## Current State
 The engine has a lot of minor inefficiencies that I was unable to fully control. One bug is that it prefers to stalemate in endgame positions, as there are not a lot of checks if No possible moves are generated. This project can serve
 as a useful structure and key component of a tried and true chess engine, but it does not posses the refinement required to be truly competitive. 

 

 
