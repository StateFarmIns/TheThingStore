# Why go to all this trouble? This seems like a lot of work.

## Story Time

In the early 2020's State Farm's Agile Team Monty Python (Modeling, Research, and Development) picked up a business model which needed to transition from legacy processes to a cloud environment.

This transition included *not just* code changes but also included process changes. This model supports many potential downstream projects and, as a result, the team needed to be able to be quickly react to dynamic and previously unknown requirements. We also needed to enable the consumers of the model to track their model development and implementation details in their new environment in a relatively efficient and flexible manner.

We wanted to do all that in an **automated** way, because doing repetitive work by hand is for the birds.

So, we developed the Thing Store. It allowed us to begin automating large portions of the work cycle management by designing our processes around a communal and shared data layer with dynamic structural expectations.

Previously, we needed at least a trained Data Scientist and Data Engineer to develop the modeling data, run the model, and enable the analysis following the modeling.

Now the only point where a Data Scientist is typically involved in the modeling flow is process research and development. They are removed from the production model development and generally only work in an 'expert contractor' style role.

Although Data Engineers are (for now...) still involved in the development of the modeling data it is typically only in a provisioning style role, where identified data is requested prior to beginning a project, retrieved, and stored / made accessible within the data layer so as to be reused in production processes.

Our intent is to also automate that.

Almost everything can be automated if you try hard enough!

Sometimes it's **very** challenging.

## What is hard about automating work?

If you think about process automation for more than a hot minute you begin to understand the daunting complexity of attempting to automate small tasks, let alone an enterprise. Nearly every single task (initially) appears unique.

There are *similarities* across many styles of tasks, but nearly every task requires some small amount of human expertise applied.

In automation, that means that almost every task requires unique solutions (code) to 'do the thing'.

Attempting to develop one system that just 'does the thing' in every case is therefore, by definition, a task that requires almost every unique solution.

You should recognize that is an infinitely complex problem. Infinite complexity is unsolvable… right?

But in reality, it is not infinitely complex.

In reality, all solutions are similar in many ways, and they even break down to a very simple (on the surface) algorithm.

1. Here is all the stuff I use. This is the Thing that I do.
2. Stepping back, I look at how information flows through all the stuff I use when I do the Thing.
3. I think critically about how the information flows and I identify which information is used, and how.
4. I begin to remove unnecessary information and identify high value information.
5. The more I understand the information flow, the more efficient my work and storage of my information can become.
6. At some point I can identify entire portions of the work which can be executed without supervision.
7. I design solutions for independent Things within the Thing that I do.
8. I implement those.
9. I test those.
10. I remember, and use, solutions that work. I might remember solutions that do not work, if there's a good reason.
11. I repeat this process ad infinitum and in recursion.

## How can that be easy?

It can be easy if we:

* Give up on caring about exactly what is inside a particular solution,
* Standardize the way we get at solution components, and
* Explicitly measure and validate the solution.

Thus, the Thing Store.

Every Thing in the Thing Store could literally be anything. It could have any of the named components of a Thing.

Any Thing in a Thing Store can be reused in automation by knowing the name of the Thing.

## What does that mean in practice?

It means processes can be designed around process pools to remove hard coded elements. This allows us to enable independent automation and reduce reliance on Data Scientists / Data Engineers for process development.

This is done by storing everything as a reusable Thing.

Modeling workflows? Stored as a Thing.

Analysis workflows? Stored as a Thing.

Outcomes of modeling workflows? Thing.

Parameters for the modeling job? Thing.

Parameters for the data job? Thing.

Artifacts produced by the model? Stored in Things.

All of these things wrapped together in a single file that describes which Things were used to create which output? Stored as a Thing.

## Why!?

Why in all of God's Green Earth would I do that!? This sounds like unnecessary madness!

Because it makes automation **near trivial**. It separates all the backend implementation and infrastructure of work in whatever language from the process implementation of picking up the bits and sewing them together. It **explicitly encodes the 'important' process bits.**

It allows you to design towards an agentic workflow.

It allows you to conduct process optimization one piece at a time.

## What the heck is an agentic workflow?

You do 'A Thing', right? You identify a problem. You identify the information which you have available which is likely to solve the problem. You load the raw data into an environment where you structure it. You investigate the feature space. You build a model. You measure the model. You investigate the hyperparameter space. You do this in an ever-tightening cycle where you dial in the scope of the data and the scope of the acceptable range of hyper-parameters. You establish a baseline model in a manner where it can run by itself and you can interrogate it. You field data coming in and ask the model to produce output from the structured data you provide. You use the outputs of those to provide intelligent automation of business decisions. You babysit that model to ensure it's humming along nicely and is still relevant.

Ok. Sounds good.

Why does it have to be you that does any piece of that? Why can't you ask an automated agent that's proven it can do it well enough to get a credible first stab at it and simply have humans in the loop make appropriate changes to the implementation in order to resolve arbitrary issues?

Every sentence above can be deconstructed and potentially run in automation by an intelligent agent. Humans often need to be embedded within this loop, but this loop, wherein an automated agent is capable of performing a significant portion of the work, is being dialed in right now.

What else does this give us?

It allows you to conduct testing of your processes, regardless of their complexity. It enables logging and intelligent automation methods to increase the efficiency and effectiveness of the storage method to support the work it enables.

It allows for investigating many individual components for semantic similarity.

It allows for teams to reuse processes.

It allows for independent and siloed process development to still be useful and reusable.

It allows for tying together many different and independent silos which all contain expert knowledge.

It allows for reusing Things across those silos.

It allows for finding Things, from outside the silos.

It allows for automated agents to curate the shape of Things such that work is more effective and efficient over time.
