# Mafia (party game) - English Wikipedia

- Source: https://en.wikipedia.org/w/index.php?title=Mafia_(party_game)&oldid=1387169334
- Revision: 1387169334 (2026-11-22)
- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), authors: Wikipedia contributors
- Retrieved: 2026-11-24. Changes: plain-text extract of selected sections only (History, Gameplay, Game theory, standard investigative/protective roles, win-condition and tied-vote variations, Artificial intelligence)

Mafia, also known as Werewolf, is a social deduction game created in 1986 by Dmitry Davidoff, then a psychology student at Moscow State University. The game models a conflict between two groups: an informed minority (the mafiosi or the werewolves) and an uninformed majority (the villagers). At the start of the game, each player is secretly assigned a role affiliated with one of these teams. The game has two alternating phases: first, a night-phase, during which those with night-killing-powers may covertly kill other players, and second, a day-phase, in which all surviving players debate and vote to eliminate a suspect. The game continues until a faction achieves its win condition; for the village, this usually means eliminating the evil minority, while for the minority, this usually means reaching numerical parity with the village and eliminating any rival evil groups.

## History
Dimitry Davidoff (Russian: Дми́трий Давы́дов, Dmitry Davydov) is generally acknowledged as the game's creator. He dates the first game of Mafia to spring 1987 at the Psychology Department of Moscow State University, from where it spread to the classrooms, dorms, and summer camps. Davidoff says that he brought Mafia into the Psychology department classrooms for research and it spread (as a meme) from there to dormitories and likely over next summer, through student summer camps. He credits this game-based methodology to pioneering 1920s psychologist Lev Vygotsky and the Turing test. He developed the game to combine psychology research with his duties teaching high school students. It became popular in other colleges and schools in the Soviet Union, often associated with hugely popular TV series La Piovra, which first ran in 1986. In the 1990s, it began to be played in other parts of Europe (in some countries under the name City of Palermo) and then the United States. By the mid 1990s, a version of the game became a Latvian television series with a parliamentary setting, and played by Latvian celebrities.
In 1997, the werewolf theme was added to the game by Andrew Plotkin (also known as Zarf), arguing that the mafia had less cultural resonance, and that the werewolf concept fit the idea of a hidden enemy who looked normal during the daytime.
The Werewolf variant of Mafia became widespread at major tech events, including the Game Developers Conference, ETech, Foo Camps, and South By Southwest. In 1998, the Kaliningrad Higher school of the Internal Affairs Ministry published the methodical textbook Nonverbal communications. Developing role-playing games 'Mafia' and 'Murderer' for a course on Visual psychodiagnostics, to teach reading body language and nonverbal signals. In September 1998, Mafia was introduced to the Graduate College at Princeton University, where several variants were developed. The werewolf theme was also incorporated in the French adaption of Mafia, The Werewolves of Millers Hollow.
In 2016, a film, Mafia: The Game of Survival was made in Russia based on the game; however, it received negative reviews.

## Gameplay

In its simplest form, Mafia is played by two teams: the mafiosi and the villagers. Live games require a moderator who does not participate as a player, and identities are assigned by handing out cards, or by other non-verbal methods such as physically tapping players. At the start of the game, every mafioso is given the identities of their teammates, whereas the innocents only receive the number of mafiosi in the game.
In an open setup, the numbers of each power role (e.g. militia) present in the game is known to the players, while in a closed setup, this information is not revealed, and in a semi-open setup, only limited or tentative information about the power roles is revealed.
There are two phases: night and day. At night, certain players secretly perform special actions; during day, players discuss and vote to eliminate one player. These phases alternate with each other until all mafiosi have been eliminated or they reach numerical parity with the innocents.
Some players may be given roles with special abilities (see below). Common special roles include an investigative role, a protector role, and some association roles.
Andrew Plotkin recommends having exactly two mafiosi, whereas the original Davidoff rules suggest a third of the players (rounding to the nearest whole number) be mafiosi. Davidoff's original game does not include roles with special abilities. In his rules for "Werewolf", Plotkin recommends that the first phase be night and that there be an odd number of players (including the moderator). These specifications avoid tie votes for eliminations and ensure that the game will end dramatically on an elimination rather than anticlimactically with murder as a foregone conclusion.

### Night

All players are told to close their eyes. The moderator then instructs all mafiosi to open their eyes and acknowledge their accomplices via tap on the shoulder. The mafiosi pick a "victim" by silently gesturing to indicate their target and to show unanimity then close their eyes again.
A similar process occurs for other roles with nightly actions. In the case of the seer, the moderator may indicate the target's innocence or guilt by using gestures such as nodding or head shaking. Night may be accompanied by players tapping gently to mask sounds made by gesturing.

### Day

The moderator instructs players to open their eyes and announces who "died" the previous night. Discussion ensues among the living players. At any point, a player may accuse someone of being a mafioso and prompt others to vote to eliminate them. If over half of the players do so, the accused person is eliminated and night begins. Otherwise, the phase continues until an elimination occurs.
According to some rules, the role of dead players should be revealed; according to others, for example, if the protector dies, nobody should know that. In both cases, dead players are not permitted to attempt to influence the remainder of the game.
Because players have more freedom to deliberate, days tend to be longer than nights.

## Game theory

### Mathematical study
Mafia is a complicated game to model, so most analyses of optimal play have assumed both (a) that there are only townsfolk and Mafiosi and (b) that the townsfolk never have a probability of identifying the Mafia that is better than chance. Early treatment of the game concentrated on simulation, while more recent studies have tried to derive closed-form equilibrium solutions for perfect play.
In 2006, the computer scientists Braverman, Etesami and Mossel proved that without detectives and with perfect players the randomized strategy is optimal for both citizens and mafia. When there is a large enough number of players, to give both groups similar probability of winning, they showed that the initial number of mafiosi m needs to be proportional to the square root of the total number of players P, that is m∝P. With a simulation, they confirmed that 50 mafiosi would have almost a 50% chance to win among 10,000. The Mafia's chance of victory is
W(m,P)≈mP,
which is a good approximation when the right hand side is below 40%. If any detectives are added to the game, Braverman et al. proved that the number of Mafiosi must remain at a fixed proportion of the total number of players for their chance of winning to remain constant.
In 2008, Erlin Yao derived specific analytical bounds for the mafia's win probability when there are no detectives.
In a paper from 2010, the exact formula for the probability that the mafia wins was found. Moreover, it was shown that the parity of the initial number of players plays an important role. In particular, when the number of mafiosi is fixed and an odd player is added to the game (and ties are resolved by coin flips), the mafia-winning chance do not drop but rise by a factor of approx. π/2 (equality in the limit of the infinite number of players).
In 2023, a modification to the strategy of Braverman et al., termed the Modified Braverman Strategy (MBS), was proposed to increase the Mafia's probability of winning. When the numbers of Mafia and citizens are equal and greater than one, the MBS allows the Mafia to coordinate their votes against a predetermined citizen if the randomized resident vote selects a Mafioso, rather than voting to eliminate the selected Mafia member. Under the original Braverman strategy, for M=C=n, the Mafia's probability of winning is 1−12n; under the modified strategy it is 1−12n+1. Thus, within the model, the modified strategy weakly dominates the original strategy. Computer simulations reported in the thesis also produced higher Mafia win probabilities under the MBS than under the original strategy across the game sizes tested, with increases ranging from 1% to 13%.

### Results in live play
In live (or videoconference) real-time play, the innocents typically win more often than game theory suggests. Several reasons for this have been advanced:

The physiological stress of sustained lying degrades the initial ability of mafioso to deceive the innocents, much more than a model of perfect play would predict, especially if the innocents can get the town emotionally involved in the game's outcome:
If you're trying to feign shock or anger, it's much harder to do over a long period. People accused of something they're trying to hide will start out feigning outrage – 'How dare you ask me that?' But that will start to change to objection rather than shock, as it's psychologically very difficult to mimic emotion.
The information revealed by the mafiosi voting patterns tells against them later in the game. One of the game's fans, Max Ventilla, has said that "If the villagers are allowed to keep a pencil and paper, they always win."
As players get more experienced, their strategic sophistication and ability to spot and use deception increases. They will typically get better at the skills needed for playing innocents faster, being villagers more often than mafiosi.
The Metagame aspect: Dimma Davidoff has said past connections will always lose to future collaborations. When playing several Mafia games with the same people, it's more helpful to be known for honesty than for deceit. Davidoff considers that so important that he thinks the advantages of playing the mafioso role honestly outweigh the disadvantages.
Laboratory results have not necessarily shown the same pattern. In a 2023 study in which players were not permitted to communicate, there was no statistically significant difference between observed and theoretically predicted win rates. The Mafia won 43 of 80 games (53.8%), compared with a theoretical win probability of 51.8% (p=0.407); excluding one nine-player session, the Mafia won 37 of 70 games (52.9%), compared with a theoretical probability of 49.2% (p=0.312).
But the Mafia can win in live play; their best chance of winning occurs when mafioso bond with their innocent neighbours and convince those neighbours to value that bond over dispassionate analysis. The game designers Salen and Zimmerman have written that the deep emergent social game play in Mafia (combined with the fear of elimination) create ideal conditions for this.

## Optional roles

These additional roles are named differently in the many versions of Mafia, for thematic flavor, or historical reasons. Also, the same role-name can have differing functions across different versions of the game. What follows is a general list of role types found in Mafia variants; since the specific names vary by milieu it must be non-exhaustive.

### Investigative roles
Players with these roles use their own night-phase to discover something about other players. Though the standard game now includes the basic Detective, these roles are optional, and games can exclude them entirely (such as the stool pigeon variant, or Davidoff's original rules).

#### Investigative roles (standard)
Detective, Seer, Commandant, Sheriff, Police, etc.

Allied with the Innocents, the Detective can detect whether a player is a mafioso. They will typically wake up, and point at one person; the Narrator will silently indicate to the Detective whether that player is Mafia or Innocent. In some versions of the game, the Detective's investigation result is announced publicly by the Narrator, for example the Detective found a Mafioso!. More commonly, no announcement is made. As with other roles, which player is the detective is not generally known, leaving anyone the option of pretending to be the Detective.
A Detective is usually included in modern games. For example, somebody is always assigned this role in all commercial card game versions, and almost all internet-based, and most face-to-face games start with at least one detective. Multiple detectives either act in separate night phases (unaware of the identities of other detectives) or work together as the police (an association role).
In the online Mafia variant Town of Salem, the Sheriff is a role which can interrogate people at night to discover their possible innocence. Generally, most Town roles appear innocent; and most roles that are considered evil appear suspicious to the Sheriff.

### Protective roles
Guardian Angel, Doctor, Bodyguard, Hero, Jailer, etc.

Allied with the Innocents, the Doctor-type role defends others at night. Typically, they will awaken at night after the Mafia have gone back to sleep and point at one person to protect; that person will survive any night-time attack. They are typically allowed to protect themselves, and are commonly barred from protecting the same target on successive nights. A Guardian Angel can only protect others. The Nurse gains the Doctor's abilities if the Doctor dies. The Jailer protects innocents just like the doctor, but also simultaneously blocks their night action. The Firefighter, or the Herbalist can protect from some night-time attacks but not others (in Werewolf, for example, they choose one person to protect with wolfsbane, but that person may still be killed by the Serial Killer). Other games limit this ability to a certain number of times.
Bodyguard-type roles sacrifice themselves to prevent (and sometimes kill the corresponding attacker) a single attack on a player.

## Variations
The naming of various roles, factions, and other elements of play is theme-dependent and has limitless variation. Common alternative themes restyle the mafia as werewolves, cultists, assassins, or witches, with other roles being renamed appropriately.
Over the years, players have created Mafia variants that include additional rules. Some of these are listed here.

### Variations on the win conditions
If there are as many mafiosi as innocents in the day-phase then a mafia victory is declared immediately, under the original Mafia rules. With the ability to deny a majority at an elimination vote, remaining mafiosi cannot be eliminated unless innocent/neutral killing roles exist. Other variants suspend this rule, and only declare the game after every member of one faction has been eliminated: this makes the game easier to explain, and to run.

### Election variants
Nominees for elimination may be allowed to make a speech in their own defense. Usually, each player must vote, can only vote once and cannot vote for themselves. But some variants have a more complicated process of selecting players to be executed. Davidoff's original 'Mafia' allowed multiple day-time executions (per day), each needing only a plurality to action.
Voting variants abound, but any elimination usually requires an absolute majority of the electorate, or votes cast. So the voting is usually not by secret ballot for multiple candidates with the highest vote count eliminated; it is more usual for the voting to be openly resolved either by:

A nomination or series of elections structured to ultimately offer a choice between two candidates, or
An option to eliminate (or not eliminate) one suspect (with a new suspect produced if the last one survives the vote).

#### Tied votes
Deadlocked elections can be resolved by lot or by killing the player with the scapegoat special role.
The special case of one mafioso and one innocent remaining can be decided randomly or be ruled a Mafia win—this is more usual in live play.

#### Optional elimination variant
The Innocents can choose not to kill anybody during the day. Although commonly unsure of Mafia identities, the Innocents are more likely to randomly kill a mafioso than are the Mafia (at night). Therefore, not eliminating anyone (even at random) will typically favor the Mafia.
However, when the number of survivors is even, No Kill may help the Innocents; for example, when three Innocents and one mafioso remain – voting for No elimination gives a 1/3 chance of killing the mafioso the next day, rather than a 1/4 chance today (assuming random elimination).

## Artificial intelligence
Werewolf is a subject of artificial intelligence research due to its unique features such as persuasion and deception. The game requires several AI technologies such as multi-agent coordination, intentional reading, and understanding of the theory of mind.
Deep learning has been used in an attempt to develop agents that can win the game as werewolves or as villagers. Regular expressions have been used to parse utterance logs for divulgence (or "coming-out" as a role) and decision information, although one difficulty has been that a statement such as "Player A is a werewolf" could be based on either the player's ability (e.g. as seer) or just speculation.
