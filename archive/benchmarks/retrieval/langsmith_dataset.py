from dotenv import load_dotenv
from langsmith import Client


load_dotenv("../../../.env")

client = Client()

# Define the examples for the dataset
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
]

# Create the dataset and examples in LangSmith
dataset_name = "WHO Disease Outbreak Q&A"

dataset = client.create_dataset(dataset_name=dataset_name)

client.create_examples(
  inputs=[{"question": q} for q, _ in examples],
  outputs=[{"answer": a} for _, a in examples],
  dataset_id=dataset.id,
)
