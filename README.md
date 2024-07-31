# agentflow
The prototype for the Agent LLM (Language Learning Model) Framework

## Overview
agentflow is designed to enhance the reasoning and retrieval capabilities of Language Learning Models (LLMs) by employing a system known as Retrieval Augmented Generation (RAG). This framework is aimed at developers looking to construct and evaluate robust LLM applications that leverage advanced retrieval techniques to augment natural language processing tasks.

## Framework

agentflow provides tools for constructing and evaluating Retrieval Augmented Generation (RAG) systems and enhancing LLM inference and retrieval capabilities. The stack consists of:

- **Language:** Python
- **Frameworks for LLMs:** OpenAI, Autogen
- **Databases:** ChromaDB
- **Frontend:** Flutter

### 📖 What is Retrieval Augmented Generation (RAG)?

In RAG, when a user query is received, relevant documents or passages are retrieved from a massive corpus, i.e., a document store. These retrieved documents are then provided as context to a generative model, which synthesizes a coherent response or answer using both the input query and the retrieved information. This approach leverages the strengths of both retrieval-based and generative systems, aiming to produce accurate and well-formed responses by drawing from vast amounts of textual data.

### 📖 What is an Agent?

An agent in the context of agentflow refers to a component or model that acts autonomously to perform tasks such as data retrieval, information processing, and interaction handling in an LLM ecosystem. Agents are designed to operate in a decentralized and coordinated manner, enhancing the system's overall performance and flexibility.

### 🚀 Workflow of agentflow

The workflow in agentflow involves several key steps:
1. **Query Reception:** Receive and interpret user queries using natural language understanding.
2. **Data Retrieval:** Employ agents to retrieve relevant information from the document store.
3. **Response Generation:** Synthesize responses based on the retrieved data and the initial query.
4. **Interaction and Feedback:** Use feedback from users to refine and improve the agents' performance.

### 🛠️ Development

- **Directory Structure:**
  - `/vagentic` Contains the prototypes of the agents.
  - `/instance` Stores instances of agent objects.
  - `/examples` Demonstrates the usage of agents within the framework.

### 🌐 Links & Resources

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Multimodal Chain-of-Thought Reasoning in Language Models]