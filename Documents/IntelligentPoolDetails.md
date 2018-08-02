# IntelligentPool

## Introduction
From now on, we will call the game billiard instead of intelligent pool because I feel like to.

The general name of those types of those games is cue sport(See [Wikipedia](https://en.wikipedia.org/wiki/Cue_sports)), and there are many of them. In general, you need to use a stick to hit the white ball so that the white ball hits the other balls and they go into the pockets. The game may look like this:

<p align="center">
    <img src="Images/IntelligentPool/BilliardGame.png" 
        alt="BilliardGame" 
        width="600" border="10" />
</p>

During the development of the materials for the Computational Intelligence in Games course, we decided to develop a whole set of examples with the billiard game, using different technologies, to showcase the concepts and power of each. The start of the examples is a simple case where the AI only need to hit the white ball once and try to make both of the red balls on desk into pockets, using MAES algorithm. The final goal is to develop a AI that can play a whole game with itself, where it should plan not just one shot but multiple shots and prevent the opponents from getting advantages, using PPO. 

In the end, at least until now, PPO is not working at all. We ended up make a even simpler case than the simple case in the beginning, with only one red ball and 4 pockets, and the game restarts after every shot.  

Here I will go through the development process, describe each example scenes, tell how to play with them, and explain why I think the billiard game does not work directly with pure PPO or supervised learning.
