from operator import itemgetter
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = getpass.getpass("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")
# llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
llm = ChatGroq(model="llama3-8b-8192")
def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_json("./src/data/database_table_descriptions.json")
    

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['table_name'] + "\n" + "Table Description:" + row['description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()
table_details_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Return the names of ALL the SQL tables that MIGHT be relevant to the user question.
                          The tables are:

                          {table_details}

                          Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""),
            ("human", "{question}")
        ]
    )

# table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
def get_tables(table_response: Table) -> List[str]:
    """
    Extracts the list of table names from a Table object.

    Args:
        table_response (Table): A Pydantic Table object containing table names.

    Returns:
        List[str]: A list of table names.
    """
    return table_response.name
try:
    structured_llm = llm.with_structured_output(Table)
    table_chain = table_details_prompt | structured_llm
    select_table = {"question": itemgetter("question"), "table_details": itemgetter("table_details")} | table_chain | get_tables
except Exception as e:
    select_table = None
    print(e)