# What *IS* The 'Thing' Store

It's a tool, an idea, and a promise.

## It is a tool

It is a **simple** tool to share *data*, *work*, and *outcomes* similarly to git.

It allows you to use *simple* commands to store, retrieve, and *inspect metadata* for anything you want to put in the box.

The Thing Store is a simple process tracking and management tool. It is designed from the ground up to work alongside and within automated modeling frameworks. It can easily integrate within processes at any point in their developmental lifecycle. 

## It is a standard

The Thing Store is an expectation of how to ask for Things you need in a standard way, or an API for putting and getting anything in a reliable and standard way.

The Thing Store application (Python app):
* Implements a standardized and simple API for getting at 'Things'.
* Exposes a distributed file-system like (potentially backed) tool into which you may store *any Thing*.
* Uses standard tooling, mechanisms, and data structure expectations.

It's an expectation that you can simply ask nicely for things that already exist like:

* 'Give me the raw data which represents this modeling dataset (`get_dataset(by_name)`)' and then have a tabular dataset available for modeling.
* 'Give me the analytic product generated which would allow me to do first order analysis (`get_artifacts(by_name)`)' and then have a collection of prefilled notebooks available for review.
* 'Give me the parameters for `this work that I do` (`get_parameters(by_name)`)' and then have the XGBoost modeling parameters available to update / change / insert into a specific workflow which runs XGBoost on a dataset.

It's an expectation that all things are immutable and versioned.

## It is a way of work

The Thing Store allows you to design workflows which rely *symbolically on the output and components of other work, both formally identified and versioned, to make *implicit and *delayed DAG structures easily.

You can chain workflows which assume they have access to a common data layer (with Things you need available in the Data Layer) and describe arbitrary work explicitly and sequentially, if you so desire, by reusing both the structure of your metadata (which is as custom as you desire), and the contents of the pool.

# It is backend independent

The basic ThingStore implements a standardized API that captures and retrieves 'Things' (Check out the docs!). The API interacts with an implemented backend to enable asking for an explicit Thing from a common data layer. Explicit Things, at that point, are *all* data sources that can be dynamically identified, stored centrally, and reused in automation.

That backend data layer can be (and is for our default implementation) a FileSystem implementation where you say, simply, 'Create a managed pool for me here'. When you turn a ThingStore on for the first time it will create default data structures at the location that you're requesting (if you have write access...) using the backend of your desire (again, FS by default).

Then, it simply records data explicitly when you ask it to, in the manner specified by the back end.

## It is a shared promise

Centralizing process data describing datasets and methods enables both human-developed and automated algorithmic enhancements within process flows. It provides a central and standard method for accessing and reusing data within a measurable framework.

The collection of data, metadata, and methods together greatly enhances the transition from unmanaged processes to human-in-the-loop to full automation as business processes formally develop and evolve.

Working within this framework allows many people with differing objectives to create, store, and reuse in a shareable and reusable way that ultimately is designed to save money and be better, faster.

The distributed, hierarchical, and shared nature of managed sets of process component nodes enables the hierarchical (and dynamic) definition of a *problem domain.

* Symbolically here means that we can reference it by name as a symbol / variable and the contents are not static. We can define workflows in terms of functions which take variables, and those variables are fileid, just like f(x) where x can take any value f(fileid) can take a fileid which can have variable contents.
* Implicit v explicit is 'not specifically written down somewhere in one place' v 'all the graph info collected in one file'.
* Delayed: I can call it to execute it.
* What is a problem domain? A problem domain can be thought of as 'all the data that I need to think about to solve my problem mathematically.'