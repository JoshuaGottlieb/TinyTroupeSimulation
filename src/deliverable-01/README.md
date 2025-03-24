# Deliverable 1

The requirements for deliverable one were 1) install the tinytroupe Python package, 2) create a persona and interact with it, and 3) set up a two-way conversation between two personas and let them run for 3 or more iterations. Proof of fulfillment of these requirements are below.

## Installation of TinyTroupe Package

Following the instructions on the TinyTroupe Github page, I used the command

> pip install git+https://github.com/microsoft/TinyTroupe.git@main

within my Linux console. This prompted the download and installation of many packages, as I had recently upgraded my machine to a new version of Python and did not have any packages pre-installed.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/tinytroupe_install-01.png)

Many packages later:

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/tinytroupe_install-02.png)

## Creation of and Interaction with a TinyPerson Persona

The creation and interaction with a TinyPerson persona can also be found in the [TinyTroupe-TestSimulation](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/TinyWorld-TestSimulation.ipynb) notebook. I created a TinyPerson named Sofia Moreau who is a 37 year old French Cybersecurity Analyst. I also created a second TinyPerson named Elias Richter who is a 52 year old German CEO of TitanTech Solutions, a fintech company who started as a software developer and is looking to expand TitanTech's secure digital banking solutions. Two personas were created, as they need to interact with each other in the next part of the deliverable.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/create_tinypeople.png)

I then interacted with the Sofia persona by asking her what new solutions in cybersecurity were the most interesting to her.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/tinyperson_interaction.png)

## Creation of a TinyWorld and Interagent Conversation

The creation of a TinyWorld and conversation between the personas can also be found in the [TinyTroupe-TestSimulation](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/TinyWorld-TestSimulation.ipynb) notebook. I created a TinyWorld to act as a conference call between Sofia and Elias. I gave Elias some context in order to kickstart the conversation, stating that he had hired Sofia to explore secure digital banking solutions and that his goal was to ask Sofia for updates on idea for improving TitanTech's cybersecurity.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/create_tinyworld.png)

In hindsight, the context given to Elias may have been slightly too broad, as Elias and Sofia end up talking about general cybersecurity ideas rather than those pertaining specifically to digital banking. If this conversation needed to be refined, the context given should be refined to direct the conversation towards a more specific topic. The world was run for 5 iterations.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/run_tinyworld-01.png)

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/run_tinyworld-02.png)

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/run_tinyworld-03.png)

Interestingly, Sophia and Elias ended their previous conversation and started up a new one after only 3 iterations. Presumably, the new conversation is set some time after the initial proposals.

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/run_tinyworld-04.png)

![](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/blob/main/src/deliverable-01/screenshots/run_tinyworld-05.png)

As the world was set to run for only 5 iterations, the conversation stops here.
