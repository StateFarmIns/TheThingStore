# Help, I need to automate everything. Right now.

Or: `How do I take this new piece of work (or potentially something which is already understood) and rigorously define it so that I can automate the stuffing out of it in a reusable manner.`

This is the first of a series of three blog-post-esque Jupyter Notebooks which provide a crash course on how to walk from zero to hero; it walks through Process Engineering, Sequence Modeling, and introduces the ThingStore, which is a **tool**, a **framework**, and a **way of mind**.

Links to ThingStore docs as appropriate here.

### Brief Introduction to Team Monty Python and the Insurance Industry with **zero** technical details.

Team Monty Python is an agile research team supporting our property and casualty modeling department; we're responsible for traditional data science work (i.e. solving a **specific** business problem) and also with researching more modern tools, technologies, tactics, and procedures.

We've developed something that can aid in growing towards automation; the rest of this walks through the thoughts you need to understand **why**.

## Process Engineering

Process engineering focuses on designing, analyzing, and improving work processes to enhance efficiency, productivity, and quality. It involves breaking down any type of work process into a sequence of operations, which can be represented visually through graphs or flowcharts.

To illustrate this concept, let's consider a production line in the post-industrial revolution era. Initially, humans performed manual tasks, such as assembling products or packaging goods. However, with the advent of automation, robots gradually replaced human labor, streamlining and accelerating the production process.

In this context, process engineering is crucial to ensuring that the automation systems are effectively carrying out the required tasks. Techniques like Lean Six Sigma are employed to measure and analyze these automated processes. Lean principles aim to eliminate waste and optimize efficiency, while Six Sigma focuses on reducing variability and defects.

By applying Lean Six Sigma methodologies, process engineers can monitor and evaluate the performance of automated systems, ensuring that they operate as intended and meet quality standards. Moreover, these techniques help identify any deviations or inefficiencies in the process, allowing for continuous improvement and cost management.

### Vendor and Solution Lock-In

Vendor lock-in refers to the situation where a customer becomes heavily dependent on a specific vendor's products or services, making it difficult to switch to alternatives. It arises due to factors such as proprietary technologies, complex integrations, and high switching costs. This dependency limits flexibility, bargaining power, and innovation.

In automated process engineering, solution lock-in occurs when an organization designs a process solution that is tightly coupled with specific technologies or tools. This restricts the ability to switch to alternative solutions or leverage emerging technologies. However, by adopting a modular and abstract approach, organizations can mitigate solution lock-in. By defining inputs and outputs rigorously and treating different aspects of the process as subprocesses, flexibility is retained in selecting different technologies for each subprocess.

While vendor lock-in is about reliance on a particular vendor's offerings, solution lock-in pertains to the dependency on a specific design or implementation of a process solution. Both types of lock-in hinder adaptability and exploration of alternatives. However, by embracing a modular and abstract approach in process engineering, organizations can mitigate the risks of solution lock-in and maintain the freedom to choose and evolve their technology solutions as needed.

## Process Operations and Automation with Large Language Models

Large Language Models can be integrated with any process graph, allowing for its execution via an API. This integration enables the utilization of the LLM's capabilities to process and analyze data within the context of the defined process. By connecting the LLM through an API, the process graph can leverage the model's language understanding, prediction, and decision-making capabilities to enhance the execution of operations.

The integration of an LLM-based API execution layer with a process graph facilitates seamless communication between the two components. The process graph defines the sequence of operations, while the LLM, accessed through the API, provides the intelligence and language processing capabilities to execute those operations effectively. This approach enables the process graph to benefit from the power of the LLM, enabling advanced data processing, decision-making, and insights generation within the context of the defined process.

Embedding humans in the loop as critical review experts allows for automation to be grown around both pre-existing and developing processes.

## Process Engineering and Data Science

In today's fluid work environments, Data Engineers (DAE), Data Scientists (DS), and Artificial Intelligence Engineers (AIE) play vital roles in leveraging data and AI technologies. DAE collect, organize, and transform data to make it accessible and usable for analysis and modeling. DS focus on applying statistical and machine learning techniques to extract insights and build predictive models. AIE focus on both application and research of modern AI techniques (generative and otherwise) within this frame of reference.

The fluid nature of these environments offers scalable infrastructure, on-demand resource allocation, and seamless integration of AI tools and services. This dynamism enables organizations to efficiently leverage advanced analytics and AI capabilities.

However, process engineering in large organizations with existing information flows faces challenges due to this fluidity. Adhering to guidelines, laws, and principles, including trackability and explainability requirements, becomes crucial. Ensuring compliance, auditability, and traceability can be time-consuming and resource-intensive. Organizations must strike a balance between the environment's agility and responsiveness and the need for robust governance and compliance frameworks.

To address these challenges, organizations should invest in effective process engineering practices that encompass proper documentation, version control, and governance mechanisms. This includes establishing data management processes, tracking data lineage, and ensuring explainability and interpretability of models. By implementing these practices, organizations can navigate the fluidity of the environment while maintaining responsiveness, compliance, and adherence to guidelines and principles.

## Modeling Sequences, and by extension, Processed

**Process modeling** involves representing a process as a graph or flow of information, enabling analysis and prediction of the next steps or outcomes. Sequence modeling is a technique used to model the sequential dependencies within a graph or time series data, allowing for predictions and forecasting. Several architectures have been developed for sequence modeling, each with its strengths and applications.

ARIMA (AutoRegressive Integrated Moving Average) is a widely used statistical model for time series forecasting. It captures both autoregressive (AR) and moving average (MA) components, making it suitable for stationary and linear data.

Exponentially Weighted Moving Average (EWMA) is another time series model that assigns exponentially decreasing weights to past observations. It is useful for capturing trends and detecting changes in data patterns.

Long Short-Term Memory (LSTM) and Recurrent Neural Networks (RNN) are deep learning models designed to capture long-term dependencies in sequential data. LSTM models excel in handling vanishing or exploding gradient problems, making them effective for tasks such as speech recognition, language translation, and sentiment analysis.

Transformers are a revolutionary architecture that has gained prominence in natural language processing tasks. They employ attention mechanisms to capture the relationships between different elements in a sequence, allowing for parallel processing and handling longer-range dependencies. Transformers have been used in large language models like GPT-4 and Gemini, which generate coherent and context-aware text by leveraging the power of massive pre-training on diverse datasets.

Modern sequence models often combine the strengths of different architectures. For example, large language models like GPT-4 and Gemini incorporate additional techniques (for example: weighted mixture of experts, where multiple models specialize in different aspects and contribute to the final prediction.) Graph Neural Networks (GNNs) are also worth mentioning, as they are designed specifically to model relationships and dependencies in graph-structured data, enabling tasks such as node classification and link prediction.

In summary, sequence modeling techniques such as ARIMA, EWMA, LSTM, RNN, and Transformers play a vital role in analyzing and predicting the next steps in a process represented as a graph or time series. Advanced models like GPT-4 and Gemini combine various architectures, while GNNs are specialized for graph-related tasks, further expanding the capabilities of sequence modeling in diverse domains.

## Rigor in Data __Science__ with the Thing Store Compliant API

Maintaining rigor in Data Science projects is essential, but it becomes increasingly challenging as projects evolve and grow in complexity. As the data sources, functionality, and data distributions change, along with the methods used to extract insights, it becomes crucial to keep track of the evolving details throughout the project's lifecycle.

### Technical solutions for explicitly codifying rigor

To address these challenges, various technical solutions are available, such as GitLab, GitHub projects, and MLFlow. These tools offer value and can be appropriate for managing and documenting processes and projects. However, when projects extend beyond an individual process, the effort required to describe and manage the work can become overwhelming. The description for a single task can take infinite forms, making it necessary to have a solution capable of describing highly complex work.

### Growth towards rigor

To tackle this complexity, one approach is to adopt a structural expectation that allows for growth towards rigor. This involves identifying and utilizing common components within processes, regardless of their specific nature. These components, referred to as "Things," provide a standardized framework for describing and managing work processes. Whether it involves fitting a model to customer data, tending to a garden, building a house, or marketing a challenging product, each process can be described using these common Thing Components.

#### What's in a Thing?

A Thing can be *anything*. Here you can see what a Thing looks like on the inside.

```python
thing: Thing = {
    'metadata': {'set': 'of', 'atomic': 'labels'},
    'parameters': {'levers': 1, 'to': 'two', 'change': 'deep'},
    'function': fn,
    'dataset': tbl,
    'embedding': tensor,
    'metrics': nums,
    'artifacts': stuff
}
```

## Closeout

There are two follow-on Notebooks.

1. Demonstration of how to use the ThingStore API (using TheThingStore of course...) to reliably automate the stuffing out of a process; what that looks like and what it enables by default.
2. Demonstration of how to take the previously generated information and apply intelligent modeling concepts to begin learning from your specific information flow in your specific processes.

