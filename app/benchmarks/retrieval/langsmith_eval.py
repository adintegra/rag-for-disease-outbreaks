from dotenv import load_dotenv
import os
from typing_extensions import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langsmith import Client
import ragas_test.rag as rag_bot

APP_PATH = "/Users/mark/Documents/Development/_Repositories/Adintegra/mas-master-thesis"

# load_dotenv("../../../.env")
load_dotenv(APP_PATH + "/.env")


client = Client()

# Define the examples for the dataset
# 7 initial general questions about disease outbreaks
# 7 specific questions about the dataset
examples = [
  (
    "Where is Ebola prevalent in Africa recently?",
    "As of the latest information available up to October 2023, Ebola outbreaks have primarily occurred in regions of Central and West Africa. Countries that have experienced recent Ebola outbreaks include the Democratic Republic of the Congo (DRC) and Uganda. These areas have seen sporadic outbreaks due to various factors, including proximity to wildlife reservoirs and the movement of infected individuals.",
  ),
  (
    "How is MERS-CoV transmitted between humans?",
    "Middle East Respiratory Syndrome Coronavirus (MERS-CoV) is primarily transmitted between humans through close contact. This typically occurs in healthcare settings or among family members and caregivers who are in close proximity to an infected person. The virus spreads via respiratory droplets produced when an infected person coughs or sneezes. Transmission can also occur by touching surfaces contaminated with these droplets and then touching the face, particularly the mouth, nose, or eyes.",
  ),
  (
    "What are the most effective preventive measures for Ebola?",
    "Effective preventive measures for Ebola include vaccination, especially in high-risk areas, which has proven to reduce transmission. In healthcare settings, strict infection control practices, such as using personal protective equipment and isolating cases, are essential. Educating communities about recognizing symptoms, seeking medical help, and practicing safe burial rituals is crucial for prevention. Surveillance and contact tracing help identify and isolate cases quickly, limiting further spread. Additionally, avoiding contact with potentially infected wildlife, such as bats, can reduce the risk of transmission.",
  ),
  (
    "What are the long-term effects of contracting Marburg virus disease?",
    "Survivors of Marburg virus disease may experience long-term effects such as chronic fatigue, joint and muscle pain, and, in some cases, vision or hearing problems. Psychological challenges, including depression, anxiety, or post-traumatic stress disorder (PTSD), can also persist. Additionally, some individuals might face lasting impacts on organ function, particularly affecting the liver or kidneys. The severity and presence of these effects can vary based on the initial severity of the disease and the level of care received.",
  ),
  (
    "What are the funding sources for outbreak response efforts?",
    "Outbreak response efforts are funded by a combination of sources, including governments, international organizations, non-governmental organizations (NGOs), and private donors. Governments allocate funds from their budgets to support public health initiatives, while international organizations like the World Health Organization (WHO) and the United Nations provide financial assistance and technical support. NGOs and private donors contribute through donations and grants to fund specific projects or response efforts. Funding may also come from research institutions, philanthropic organizations, and public-private partnerships.",
  ),
  (
    "What treatments are available for Yellow Fever?",
    "Currently, there is no specific antiviral treatment for yellow fever, so supportive care is the main approach. This care may include hospitalization, ensuring adequate hydration, and managing symptoms such as fever, pain, or organ dysfunction. In severe cases, intensive care may be required to support vital organ functions. Prevention through vaccination is highly effective, providing long-term protection against the disease.",
  ),
  (
    "What role do social media platforms play in tracking outbreak reports?",
    "Social media platforms play a significant role in tracking outbreak reports by providing real-time data on disease trends, public sentiment, and misinformation. Health authorities and researchers monitor social media to identify potential outbreaks, track disease spread, and assess public perceptions and behaviors. Platforms like Twitter, Facebook, and online forums can help identify emerging health threats, dispel rumors, and engage with communities to promote accurate information and public health measures.",
  ),
  (
    "Is Zika virus prevalent in the European region?",
    "Zika virus is not considered endemic in the European region, but sporadic cases have been reported in France and The Netherlands in 2016 and 2019.",
  ),
  (
    "How many Malaria cases were reported in Ethiopia between January and October 2024?",
    "Between 1 January and 20 October 2024, over 7.3 million malaria cases and 1157 deaths (CFR 0.02%) were reported in Ethiopia.",
  ),
  (
    "Which African countries are at risk of a Malaria outbreak in 2024?",
    "For 2024, WHO assesses the regional risk as moderate due to concurrent malaria and other vector-borne disease outbreaks in Ethiopia and six neighbouring countries: Djibouti, Eritrea, Kenya, Somalia, South Sudan, Sudan.",
  ),
  (
    "Was there a Chikungunya outbreak in the Kenya in 2016?",
    "Yes, there was a Chikungunya outbreak in Kenya in 2016. On 28 May 2016, the Ministry of Health of Kenya notified WHO of an outbreak of Chikungunya in Mandera East sub-county. As of 30 June 2016, 1,792 cases had been reported.",
  ),
  (
    "Is the global incidence of Cholera increasing or decreasing?",
    "Since the first disease outbreak news on the global cholera situation was published on 16 December 2022, the global situation has further deteriorated with additional countries reporting cases and outbreaks. Since mid-2021, the world is facing an acute upsurge of the 7th cholera pandemic characterized by the number, size and concurrence of multiple outbreaks, the spread to areas free of cholera for decades and alarming high mortality rates.",
  ),
  (
    "Which countries in the African region is Mpox prevalent in?",
    "Burundi, Kenya, Rwanda, Uganda, Côte d'Ivoire are some of the countries in the African region where Mpox is prevalent. The current expansion of mpox in the African continent is unprecedented. At least four countries have identified cases for the first time and others, such as Côte d’Ivoire, are reporting re-emerging outbreaks.",
  ),
  (
    "What are the symptoms of Mpox?",
    "Mpox causes signs and symptoms which usually begin within a week of exposure but can start one to 21 days later. Symptoms typically last for two to four weeks but may last longer in someone with a weakened immune system. Normally, fever, muscle aches and sore throat appear first, followed by skin and mucosal rash. Lymphadenopathy (swollen lymph nodes) is also a typical feature of mpox, present in most cases.",
  ),
]

# Create the dataset and examples in LangSmith
dataset_name = "WHO Disease Outbreak Q&A II"

# dataset = client.create_dataset(dataset_name=dataset_name)

# client.create_examples(
#   inputs=[{"question": q} for q, _ in examples],
#   outputs=[{"answer": a} for _, a in examples],
#   dataset_id=dataset.id,
# )

# For loading existing datasets
dataset = client.read_dataset(dataset_name=dataset_name)


# Grade output schema
class CorrectnessGrade(TypedDict):
  # Note that the order in the fields are defined is the order in which the model will generate them.
  # It is useful to put explanations before responses because it forces the model to think through
  # its final response before generating it:
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


# Grade prompt
correctness_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOllama(
  model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0
).with_structured_output(CorrectnessGrade, method="json_schema")

# grader_llm = ChatGroq(
#   model="llama-3.3-70b-versatile", temperature=0
# ).with_structured_output(CorrectnessGrade, method="json_mode")


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
  """An evaluator for RAG answer accuracy"""
  answers = f"""      QUESTION: {inputs["question"]}
GROUND TRUTH ANSWER: {reference_outputs["answer"]}
STUDENT ANSWER: {outputs["answer"]}"""

  # Run evaluator
  grade = grader_llm.invoke(
    [
      {"role": "system", "content": correctness_instructions},
      {"role": "user", "content": answers},
    ]
  )
  return grade["correct"]


# Grade output schema
class RelevanceGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  relevant: Annotated[
    bool, ..., "Provide the score on whether the answer addresses the question"
  ]


# Grade prompt
relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOllama(
  model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0
).with_structured_output(RelevanceGrade, method="json_schema")

# relevance_llm = ChatGroq(
#   model="llama-3.3-70b-versatile", temperature=0
# ).with_structured_output(RelevanceGrade, method="json_mode")


# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
  """A simple evaluator for RAG answer helpfulness."""
  answer = f"""      QUESTION: {inputs["question"]}
STUDENT ANSWER: {outputs["answer"]}"""
  grade = relevance_llm.invoke(
    [
      {"role": "system", "content": relevance_instructions},
      {"role": "user", "content": answer},
    ]
  )
  return grade["relevant"]


# Grade output schema
class GroundedGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  grounded: Annotated[
    bool, ..., "Provide the score on if the answer hallucinates from the documents"
  ]


# Grade prompt
grounded_instructions = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatOllama(
  model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0
).with_structured_output(GroundedGrade, method="json_schema")

# grounded_llm = ChatGroq(
#   model="llama-3.3-70b-versatile", temperature=0
# ).with_structured_output(GroundedGrade, method="json_mode")


# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
  """A simple evaluator for RAG answer groundedness."""
  doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
  answer = f"""      FACTS: {doc_string}
STUDENT ANSWER: {outputs["answer"]}"""
  grade = grounded_llm.invoke(
    [
      {"role": "system", "content": grounded_instructions},
      {"role": "user", "content": answer},
    ]
  )
  return grade["grounded"]


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
  explanation: Annotated[str, ..., "Explain your reasoning for the score"]
  relevant: Annotated[
    bool,
    ...,
    "True if the retrieved documents are relevant to the question, False otherwise",
  ]


# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a set of FACTS provided by the student.

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOllama(
  model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema")

# retrieval_relevance_llm = ChatGroq(
#   model="llama-3.3-70b-versatile", temperature=0
# ).with_structured_output(RetrievalRelevanceGrade, method="json_mode")


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
  """An evaluator for document relevance"""
  doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
  answer = f"""      FACTS: {doc_string}
QUESTION: {inputs["question"]}"""

  # Run evaluator
  grade = retrieval_relevance_llm.invoke(
    [
      {"role": "system", "content": retrieval_relevance_instructions},
      {"role": "user", "content": answer},
    ]
  )
  return grade["relevant"]


def target(inputs: dict) -> dict:
  response = rag_bot.ask(inputs["question"])
  return response


experiment_results = client.evaluate(
  target,
  data=dataset_name,
  evaluators=[correctness, groundedness, relevance, retrieval_relevance],
  experiment_prefix="rag-doc-relevance",
  metadata={"version": "Nomic embeddings, phi-4"},
)

# metadata={"version": "Nomic embeddings, llama3.2-3B"},
# print(target({"question": "Where is Dengue prevalent in Africa?"}))

# Explore results locally as a dataframe if you have pandas installed
# experiment_results.to_pandas()
