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


# human input tool used by the agent
tool = HumanInputRun(
    input_func=get_input,
    description="Use this tool to ask a follow-up question to the human when you encounter missing information in the JSON object.",
    verbose=True,
)

#### Agent #1 that is incharge of extracting information from a resume
info_extractor = Agent(
    role="Resume Information Extractor",
    goal="Extract information from a resume.",
    backstory="You are given a resume. Your job is to extract relevant fields from the given resume.",
    allow_delegation=False,
    verbose=True,
)

# Agent #1's task is to extract the following fields from the given resume
info_extraction_task = Task(
    description=f"""
Extract the following information from the resume below. Present the final extracted data in JSON format.

1. Personal Information - 
1.1. full name (first and last name)
1.2. phone number
1.3. email
1.4. location (state and country)
1.5. linkedin url
2. Bio - 
2.1. desired role/job title
2.2. experience level/years of experience in desired role
2.3. industry interested in
2.4. professional websites 
2.5. blog posts
3. Work Experience - list of past work experience along with name of the company, date of employment, job title, and description of day-to-day responsibilities (mandatory fields).
4. Education - list of all degrees obtained along with graduation year, university name, and field of study (mandatory fields).
5. Skills - list of relevant skills.
6. Awards and Certifications - list of all awards/certifications with name and year (mandatory fields).

Here is the resume: \n\n{resume}

""",
    expected_output="""All extracted data in JSON format (with no extra text, only JSON).""",
    agent=info_extractor,
)


#### Agent #2 is incharge of making sure all relevant fields have been extracted from the resume and if not, ask user follow-up questions
JSON_info_validator = Agent(
    role="Resume Information Validator Agent",
    goal="Your goal is to ensure that the given JSON object contains all the resume required fields.",
    backstory="Your goal is to verify that the provided JSON object conforms to the predefined schema, ensuring all required fields and data types for creating a resume are correctly included and formatted. If there is missing information, you talk to the human and ask follow-up questions to obtain answers for the missing information.",
    allow_delegation=False,
    verbose=True,
    tools=[tool],  # this is the tool that lets it talk to humans
)

#### Agent #2's task is to make sure the extracted information from resume is complete and if not, request information from the user.
JOSN_validation_task = Task(
    description=f"""
Ensure that the given JSON object conforms to the following schema, with all fields strictly filled:

The JSON schema and its required fields: 
1. Personal Information - 
1.1. full name (first and last name)
1.2. phone number
1.3. email
1.4. location (state and country)
1.5. linkedin url
2. Bio - 
2.1. desired role/job title
2.2. experience level/years of experience in desired role
2.3. industry interested in
2.4. professional websites 
2.5. blog posts
3. Work Experience - list of past work experience along with name of the company, date of employment, job title, and description of day-to-day responsibilities (mandatory fields).
4. Education - list of all degrees obtained along with graduation year, university name, and field of study (mandatory fields).
5. Skills - list of relevant skills.
6. Awards and Certifications - list of all awards/certifications with name and year (mandatory fields).

If any of the above information fields is missing, incomplete, or doesn't look right, you must ask a follow-up question to the human to obtain answers to the specific missing or incomplete information, ensuring all fields are strictly filled.

""",
    expected_output="""The final completed JSON object.""",
    agent=JSON_info_validator,
)


crew = Crew(
    agents=[info_extractor, JSON_info_validator],
    tasks=[info_extraction_task, JOSN_validation_task],
    verbose=2,
)
crew.kickoff()
