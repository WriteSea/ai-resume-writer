from crewai import Agent, Task, Crew
from langchain_community.tools import HumanInputRun

import os
from PyPDF2 import PdfReader


## llama 3 8b instruct model writesea's endpoint
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_MODEL_NAME"] = "meta-llama/Meta-Llama-3-8B-Instruct"
os.environ["OPENAI_API_BASE"] = ""

#### openai gpt4
# os.environ["OPENAI_API_KEY"] = ""


# pdf parser for parsing resumes
def parse_pdf(pdf_name):
    try:
        reader = PdfReader(pdf_name)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        # raise_error("Error parsing pdf, invalid file!", status_code=400)(edited)

    return text


resume = parse_pdf(pdf_name="julia_jose.pdf")


# this function is used by the agent to get human input
def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


# human input tools used by the agents
tool = HumanInputRun(
    input_func=get_input,
    description="Use this tool to ask a follow-up question to the human when you encounter missing information in the JSON object.",
    verbose=True,
)
writer_tool = HumanInputRun(
    input_func=get_input,
    description="Use this tool to ask specific follow-up questions to the human while transforming (re-writing) the resume.",
    verbose=True,
)


############################################################################################
# ---------------------------------------AGENT 1---------------------------------------------
############################################################################################

## Agent #1 that is incharge of converting text resume to JSON format
resume_extractor = Agent(
    role="Resume Information Extractor",
    goal="Extract information from a resume.",
    backstory="You are given a resume. Your job is to extract relevant fields from the given resume.",
    allow_delegation=False,
    verbose=True,
)

## Agent #1's only task is to take text resume and convert to JSON (with predefined schema)
resume_JSON_convertion = Task(
    description="""
Extract the following information from the resume given below. Present the final extracted data in the following JSON format. Ensure no details from the resume are missed during extraction. 

{
  "personalInformation": {
    "firstName": "string",
    "lastName": "string",
    "phoneNumber": "string",
    "email": "email",
    "location": {
      "state": "string",
      "country": "string"
    },
    "linkedinURL": "url",
    "professionalWebsites": ["url"]
  },
  "workExperience": [
    {
      "companyName": "string",
      "dateOfEmployment": "date",
      "jobTitle": "string",
      "responsibilities_description": ["string"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "graduationYear": "date",
      "universityName": "string",
      "fieldOfStudy": "string"
    }
  ],
  "skills": ["string"] or ["json"],
  "awardsAndCertifications": [
    {
      "name": "string",
      "issuingAuthority": "string",
      "yearIssued": "integer"
    }
  ]
}

"""
    + f"""Here is the resume: \n\n{resume}

""",
    expected_output="""All extracted data in JSON format.""",
    agent=resume_extractor,
)


############################################################################################
# ---------------------------------------AGENT 2---------------------------------------------
############################################################################################

## Agent #2 is in charge of gathering missing information
resume_info_gatherer = Agent(
    role="JSON Completion Specialist",
    goal="Ensure the given JSON object is fully completed with no missing information.",
    backstory="You specialize in verifying the completeness of JSON objects. When you find missing information, you interact with the human to ask specific questions that help gather the missing data.",
    allow_delegation=False,
    verbose=True,
    max_iter=30,
    tools=[tool],
)

# --------------- Agent 2 Task 1 ---------------
## Request missing information from user
obtain_missing_info = Task(
    description="""
Identify fields with missing values in the given JSON object.
For each field that is missing a value, ask the human a specific follow-up question to gather the required missing information.
Continue this process until all fields are fully completed. Return the fully completed JSON object.
""",
    expected_output="""A fully completed JSON object with all missing values filled in.""",
    agent=resume_info_gatherer,
)

# --------------- Agent 2 Task 2 ---------------
## Obtain career goals information from user
obtain_career_goals = Task(
    description="""
Engage with the human to obtain the following career-related information by asking specific questions one by one.
"careerGoals": {
      "type": "object",
      "properties": {
        "desiredRole": {
          "type": "string",
          "description": "Desired role/job title"
        },
        "experienceLevel": {
          "type": "string",
          "description": "Experience level or years of experience in desired role"
        },
        "industryInterestedIn": {
          "type": "string",
          "description": "Industry interested in"
        },
        "exampleJobPosting": {
          "type": "string",
          "description": "Description of an example job posting you are interested in"
        }
      }

Finally, return the inital JSON object along with this appended "careerGoals" goals field.
""",
    expected_output="""The fully completed JSON object with the "careerGoals" field appended to intial JSON object.""",
    agent=resume_info_gatherer,
)


############################################################################################
# ---------------------------------------AGENT 3---------------------------------------------
############################################################################################

## Agent #3 is incharge of drafting an impactful resume
resume_writer = Agent(
    role="Resume Writer Agent",
    goal="Draft an impactful resume",
    backstory="Your goal is to transform the given JSON information into a compelling resume. You will perform various tasks, including using job descriptions to help identify keywords and modifying the skills section of the resume to match them. Additionally, you will convert work experience entries to the CAR (Challenge, Action, Result) format for maximum impact.",
    allow_delegation=False,
    verbose=True,
    tools=[writer_tool],
)

# --------------- Agent 3 Task 1 ---------------
## Agent #3's task 1 is to transform the work experience section to use CAR
transform_workExp = Task(
    description=f"""
You are provided a JSON object with a 'workExperience' field that contains a list of JSON objects corresponding to the human's employment details. For each listed work experience JSON object, you must transform (re-write) the corresponding responsibilities field to tell a compelling story using the CAR format (Challenges, Action, Result).

Follow these steps for each listed 'workExperience' JSON object:
1. Analyze the given responsibilities field and extract CAR information.
2. Interact with the human and ask them to confirm if the extracted CAR information is correct. Ask them very specific questions to gather any missing required information.
3. Replace the existing responsibilities field with the extracted CAR information.

Finally return the inital JSON object with the updated 'workExperience' field.

    """,
    expected_output="""The full initial JSON object with updated "workExperience" field.""",
    agent=resume_writer,
)

# --------------- Agent 3 Task 2 ---------------
## Agent #3's task 2 is to transform the skills section based on desired job posting description
transform_skills = Task(
    description=f"""
Your task is to extract keywords from the value provided in the 'exampleJobPosting' field in the given JSON object. Then, compare these extracted keywords with the entries in the ;'skills' field of the JSON object. For any keyword that is not present in the 'skills' field, interact with the human to ask if they have experience with that specific skill. If the human confirms they have experience with the skill, add it to the 'skills' field. Finally, return the updated JSON object with the modified 'skills' field.

For example:

1. Extract keywords (e.g., Python, SQL) from exampleJobPosting.
2. Compare these keywords with the skills field.
3. If a keyword like 'Python' is missing, ask the human 'Do you have experience with Python?'
4. If the human confirms, add 'Python' to the skills field.
5. Return the full updated JSON object.

    """,
    expected_output="""The full initial JSON object with updated "skills" field.""",
    agent=resume_writer,
)


#### the crew
crew = Crew(
    agents=[resume_extractor, resume_info_gatherer, resume_writer],
    tasks=[
        resume_JSON_convertion,  # convert text resume to json resume
        obtain_missing_info,  # obtain missing information
        obtain_career_goals,  # understand career goals
        transform_workExp,  # transform work experience section
        transform_skills,  # transform skills section
    ],
    verbose=2,
)
crew.kickoff()
