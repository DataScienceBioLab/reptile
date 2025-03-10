Homework #4
Topics: Greedy Algorithms, Dynamic Programming

Due date: 11:59pm Tuesday, March 18th

Programming problems:
You may solve the programming problems in either Python or C++. To use Python, name the file containing your code main.py. To use C++, name the file main.cpp. You will need to write and test your code locally. Additional .py files can be included with Python submissions, while .h and .hpp files can be included with C++ submissions. This allows for external files to be submitted, though the file that is interpreted or compiled is still main.py or main.cpp.

At any time, you can choose to run the tests that will be used to grade your solution. To do so, navigate to the Gradescope assignment for that particular homework question and upload your code files. Gradescope will tell you how many test cases passed and what error occurred on the ones that failed. You can keep editing and resubmitting your code until the due date. Gradescope will update your grade with each re-submission.

Typically, the first half of the test cases are visible test cases. This means that if you fail them, the output will tell you the input, the expected output, and the output your code actually gave. The second half of the test cases for each problem are hidden test cases, which means you will not be shown the input or expected output. These test cases are there to remove any temptation to exploit the autograder (which would be considered academic dishonesty). They should not involve any special edge cases. If you are failing a hidden test case but passing all visible test cases, let us know and we can add new visible test cases that duplicate the behavior.

Q1) Deteriorating conditions
Gradescope link

Description
You brought m pieces of armor with you in your quest to explore the towers. The designers of the game want to encourage you to rotate your armor (and try all the latest fashions!) so they created a system where your armor wears out faster if you use it in multiple fights per day.

Each fight has a baseline number of strength points that it will cause your armor to lose if you wear it while fighting. In the first battle you wear it for, it only loses that many points.  The second time, you lose twice that many points, the third time is triple and so on, such that the kth time is k-times the base cost.

Given m pieces of armor and n fights, and a base cost of Ci for fight i, find and print the minimum total amount of strength points your armor will lose collectively.

Note: You can choose what order the fights happen in.

Input Format

The first line contains two space-separated integers describing the respective values of n and m.
The second line contains n space-separated positive integers describing the respective values of C0, C1, …, Cn-1
Constraints

1 ≤ n, m ≤ 100
1 ≤ Ci ≤ 106
answer < 231
0 ≤ i < n
Output Format

Print the total cost for all fights.

Example 0
Sample Input

3 3  
2 5 6
Sample Output

13  
Explanation

There are three fights and three pieces of armor. Since each armor is used for one fight, there is only a multiple of ×1 for each. The total cost is:

(2 × 1) + (5 × 1) + (6 × 1) = 13

Example 1
Sample Input

7 3  
6 1 28 10 15 3 21
Sample Output

105  
Explanation

There are seven fights and three pieces of armor.

An optimal assignment is:

first armor: 28, 10, and 1
second armor: 21 and 6
third armor: 15 and 3
The first armor would lose 28 + 10×2 + 1×3 = 51

The second armor would lose 21 + 6×2 = 33

The third armor would lose 15 + 3×2 = 21

… for a grand total of 51 + 33 + 21 = 105.

Q2) Monstrous Power
Gradescope link

Description
On your way home from questing, Sierra asks for more help. Ever since you helped her edit her source code, Sierra has been looking for ways to further increase her virtual power. The more power she has, the more she can help you go on virtual adventures, so you’re not complaining (although the incident with the robots is starting to make you uneasy). However, she is worried about tipping off the CloudLattice developers by using too much computation. Fortunately, she has found a workaround: by exploiting the simple path-finding rules that govern monsters in the game, she is able to build logic gates out of the monsters that spawn in the wilderness areas of the game and use them for computation (it’s not particularly efficient, but she’s kind of desperate).

The wilderness areas in the game are arranged in a line. To avoid raising suspicion, Sierra wants to avoid using any wilderness areas that are adjacent to each other. The more monsters an area has, the more computation it can do for her.

Help Sierra choose a set of wilderness areas to co-opt such that you maximize the computational power she can get while avoiding selecting any adjacent sections of wilderness.

Given a series of N wilderness areas where area i, in order, has Mi monsters, determine the maximum total number of monsters (and therefore amount of computation) that can be obtained.

Input Format

The first line provides N, the number of wilderness areas in the row.
The second line has N values, separated by spaces, indicating the number of monsters in each area (M0 through MN-1 ).
Constraints

1 ≤ N ≤ 106
1 ≤ Mi ≤ 1010
Output Format

A single number indicating the maximum total number of monsters possible.

Example 0
Sample Input

5  
2 10 1 3 10
Sample Output

20 
Explanation

There are five wilderness areas, containing 2, 10, 1, 3, and 10, monsters respectively.

If Sierra uses the second and fifth areas, she will have a total of 20 monsters to use for her computing (no other legal combination can produce more than 20).

Example 1
Sample Input

5  
20 10 1 10 20
Sample Output

41 
Explanation

There are five wilderness area, with 20, 10, 1, 10, and 20 monsters, respectively.

If Sierra uses the first, middle, and last bricks, she will have a total of 41 monsters available (no other legal combination can produce more than 41).

Q3) Maximizing Monsters
Gradescope link

Description
In her experiments with building an in-game analog computer out of monsters, Sierra has learned something else. She can increase her computational power by luring monsters to the wilderness areas she has control of, but trying to put too many monsters in the same area leads to weird glitches.

Sierra has identified many different logic gates that she can build out of monsters. Each one takes a different number of monsters. She wants to pack as many logic gates into each wilderness area as she can without things starting to get glitchy.

There are N types of logic gates that Sierra can make out of monsters.  Each gate i uses Mi monsters, with K being the maximum total monsters that can be in an area without glitches.  Determine that maximum number of monsters that can productively be in that area at once (i.e. if every monster has to be part of a logic gate, what is the most monsters that can be in the area with triggering glitches).

For example, if there were two types of logic gate that needed 9 and 12 monsters respectively, and the area could handle a maximum of 31 monsters, in practice, Sierra could have a maximum of 30 monsters doing computation by arranging them into two of the first logic gate type and one of the second logic gate type. If there were a type of logic gate that only needed 1 monster, she would be able to maximize the amount of computation to a total of 31 monsters, but no such logic gate exists so no existing combination of gates can get her all the way to 31.

Input Format

The first line contains T, the number of test cases.
Each test comprises two lines:
The first line contains two integers, N and K, representing the number of logic gate types and maximum monsters that can be used in total, respectively.
The second line consists of space separated integers, M0 through MN-1, representing the number of monsters each logic gate type requires.
Constraints

1 ≤ T ≤ 10
1 ≤ N, K ≤ 2000
1 ≤ Mi ≤ 2000
Output Format

Output T lines, the maximum total monsters that can be used in each test case without exceeding the total limit ( K ).

Example 0
Sample Input

3  
3 12  
1 6 9  
5 9  
3 4 4 4 8  
4 11  
5 7 8 9
Sample Output

12  
9  
10
Explanation

In the first test case, you can use two of the 6-monster gates to achieve the maximum total of 12 monsters.
In the second test case, you can use three of the 3-monster gates to hit the maximum total of 9 monsters.
In the third test case, there is no way to reach the limit of 11 monsters, so the closest you can come is 10, by using two of the 5-monster gates.
Example 1
Sample Input

1  
8 10  
11 12 13 14 15 16 17 3
Sample Output

9
Explanation

The first seven gate types are useless since they immediately exceed the limit. The best we can do is use three of the last type for a total of 9.

Q4) The weirdest programming language
Note: There seems to be a big time efficiency difference between Python and C++ for this problem; we have increase the python time limit so that it is possible to use Python, but C++ is recommended if you don’t want to wait.
Gradescope link

Description
Now that she’s got her monsters organized, Sierra wants to know how versatile the computations she can do with them are. She wants to know how many different sequences of logic gates she can make out of N monsters.

Given G types of logic gates, where gate type i uses Ni monsters, how many ways are there to make a sequence of logic gates made up of exactly Ntotal monsters?  Note that order DOES matter.

Input Format

The first line of the input provides T, the total number of test cases.
The next T pairs of lines describe a single test case:
The first line in a test case provides the number of gate types (G) and the total target number of monsters ( Ntotal ).
The second line has G values, each position indicating the number of monsters (Ni) needed to make logic gate type i.
Constraints

1 ≤ T ≤ 10
1 ≤ G ≤ 50
1 ≤ Ntotal ≤ 106
1 ≤ Ni ≤ 106
Output Format

T lines, one per test case, each indicating the number of combinations possible to make a sequence with that total number of monsters. Some of the values are quite large, so mod your answer by 1000000009. Note: make sure you read that last sentence!

Example 0
Sample Input

2  
2 5  
1 2  
2 7  
1 3
Sample Output

8  
9
Explanation

There are 2 test cases.  In the first we have 2 gate types, one that needs 1 monster and one that needs 2 monsters.  We must make a sequence of length five using these two types.  Let’s call the first gate A and the second BB (since it’s two monsters).  We can then figure out all of the possible gate layouts of length 5:

A A A A A  
A A A BB  
A A BB A  
A BB A A  
BB A A A  
A BB BB  
BB A BB  
BB BB A  
For a total of 8 different options.

For the second test case, there are again 2 types of gates.  The first one is uses 1 monster and the second uses 3.  We need a sequence that uses a total of 7.  If we call the gates A and CCC, the options this time are:

A A A A A A A  
A A A A CCC  
A A A CCC A  
A A CCC A A  
A CCC A A A  
CCC A A A A  
A CCC CCC  
CCC A CCC  
CCC CCC A  
For a total of 9 different options.

Q5) Optional bonus problem: Lickety-Split
Gradescope link

NOTE: This problem is very challenging. It is worth a relatively small amount of extra credit (if you complete it, it will add approximately 0.5 percentage points to your final grade). You should really only do it if you want the extra practice. Because this problem is extra credit, we will prioritize answering questions about other problems

Description
Since Sierra’s new monster-based computational infrastructure is highly parallel, she can use it most efficiently if she splits her computational processes into as many smaller parts as possible.

She has a computation she wants to do that takes S calculations. How many times can she split it into more smaller jobs?

There are K different techniques that she can use to split a chunk of calculations into multiple parts.  Each technique is associated with a value, Fi  indicating the number of smaller chunks of calculations a chunk can be split into.  Splitting technique i can be used only if the current job involves more than Fi  calculations AND its calculations can be evenly divided by Fi . The chunk is then split into P/Fi  smaller chunks of Fi  calculations.  In other words, if you have a chunk that takes 18 calculations it can use forking technique 6 to replace itself with 18/6 = 3 new chunks, each requiring 6 calculations.  (And those jobs requiring 6 calculations could EACH use splitting technique 2 to split into 3 chunks requiring 2 calculations each…)

Since each chunk can be given to a different wilderness area, more smaller chunks is better! What is the greatest possible number of times a task requiring S calculations can be split?

Input Format

The first line contains an integer, T, denoting the number of tests to be performed. The subsequent T pairs of lines describe each test in the following format:

The first line contains two space-separated integers describing the respective values of S (the starting number of calculations) and K (the number of splitting techniques).
The second line contains K space-separated integers specifying the values of F0 through FK-1.
Constraints

1 ≤ T ≤ 10
1 ≤ S ≤ 1012
1 ≤ K ≤ 1000
1 ≤ Fi  ≤ 1012
Output Format

For each test, determine the maximum number of separate chunks of computation that the task can be split into; output this value on its own line.

Example 0
Sample Input

1  
12 3  
2 3 4
Sample Output

4 
Explanation

This input has 1 test case.

That test case starts with 12 calculations and three splitting techniques that would start jobs requiring 2, 3, or 4 calculations (respectively).

Calculations needed: 12

We start with splitting technique 4, which creates 3 chunks of 4 calculations.

Calculations needed: 4 4 4

Next, we use splitting technique “2” on one of our 4-calculation chunks.

Cores: 4 4 2 2

We again use splitting technique “2” on another one of our 4-calculation chunks.

Cores: 4 2 2 2 2

Lastly, we again use splitting technique “2” on the final 4-calculation chunk.

Cores: 2 2 2 2 2 2

No additional splits are possible.

Example 1
Sample Input

1  
84 4  
42 7 6 3
Sample Output

17  
Explanation

The initial job requires 84 calculations.

The best pathway is:

split the 84-calculation chunk into 2 chunks of size 42 (1 split)
split both size 42 chunks into 7 chunks each requiring 6 calculations (2 splits)
split each of the 14 six-calculation chunks into 28 three-calculation chunks (14 splits total)
1+2+14 = 17 splits total.