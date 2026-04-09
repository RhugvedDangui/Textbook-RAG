"""
RAG application — retrieves textbook content from ChromaDB and summarizes it using Ollama.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3
    3. Set SEARCH_QUERY below, then run: python app.py
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── ✏️  PUT YOUR SEARCH TOPIC HERE ───────────────────────────────────────────
SEARCH_QUERY = "Good evening everyone, this is Vishwari Shali. In previous sessions, we discussed about software development life cycle and different software models with examples. I have mentioned complete software engineering subject playlist link in below description box. Now, in today's session, we will discuss about the next important model that is spiral model. Let's start the session. In today's session, we will discuss about introduction phases, when to use spiral model and their advantages and disadvantages. Let's see all these points one by one. Now, the first point is what exactly spiral model? The spiral model initially developed by the scientist Bohem in 1986. Spiral model which is also called as meta model and spiral model generally used in large project and the project which have lots of risk and the problem. This is a main area where spiral model have used. That's why this spiral model is also called as risk driven software development process model. Spiral model is a combination of waterfall, iterative and prototyping model. We already discussed these three models in detail in previous sessions. From waterfall model, it take a step by step development approach. From iterative model, it take a customer feedback taken approach and from prototyping model, it take first developed prototype and then actual development have started this kind of approach. Next spiral model generally used in different gaming industry. For online gaming, it required lots of customer interaction and lots of risk is there. Next in Microsoft and operating system versions. For example, Windows 7, Windows 8 and after that Windows 10, 11, so there is an incremental approach. This solve disadvantages and risk in previous approach from the next versions. So in this particular field, spiral model have used. See here, in this particular diagram, spiral model had divided into the four parts. First planning, second risk analysis, third engineering and execution and fourth one evaluation. Let discuss in detail. Now see here, this is an actual structure diagram of spiral model. Here, this particular line indicate a spiral or iterations and it start with this inner point. This is a starting point and this starting point indicate the first phase that is planning phase. In planning phase, there is a communication between customers and the project head. Project head collect all the requirements from the customer, what is the need of customer and what exactly customer want. After that system analyze, analyze all these requirements and they decide estimated cost, schedule and required resources of the project. Now after collecting all these requirements in the planning phase, it move to next that is risk analysis phase. This is the second phase. See, spiral model generally used in large projects like Istro, NASA. So in that particular phase, lots of risk problems and security related issues is there. Right, that's why risk analysis is the most important phase in spiral model. Here, the first thing is they identify all the potential risks. The risk related to the technical risk, software and hardware related risk or there is a risk in problem statement in data. So they collect all these types of risk. Next, they find out the solution for solving this kind of risk or the problem. So this solution is planned in risk mitigation strategy documentation. After that, they develop a prototype. So prototype also used for solving the risk before starting the actual development. First, they develop the prototype. Prototype adjust a replica of your actual software development. So here, they develop a prototype. Next. Developing a prototype, it move to next phase that is engineering and execution. So here, actual development have started. First, designer design the product as per the final prototype or the requirement. Developer perform actual coding, biusing different programming languages, tester perform, testing, biusing different testing methods. They check whether the project is related to the customer requirements or not. And last one, after design coding and testing, they deploy the product to the customer environment. So this is the third part. Now, after deploy, it move to next one that is customer evaluation. So in this fourth part, they take a feedback from customer. And if customer want any changes, it move to next spiral iteration. Means it move to next one that is again planning, again risk analysis, again, engineering and execution and again, they take a evaluation. Means second spiral iteration is there. Again, they check if there are customer want any suggestions or customer have any needs. So they again move to planning, risk analysis, engineering and execution and evaluation in this way. Means if customer want any changes, it move to next spiral iteration. So the spiral iteration increases, means cost of the project increases. And spiral iteration, it just one iteration is there. Means after one iteration, customer doesn't want any changes. Customer are satisfied. Means their project cost is less clear. So this is called a spiral model. Now, when to use spiral model. First, when the project is large and high project project is there, second here customer requirements are unclear and complex customer requirements. That's why customer requirements are continuously changing is there. The important thing is here risk evaluation is there. Suppose the particular project having lots of risk. So at that time spiral model have used here, they create a prototype for solving the risk problem. And also there is a no deadline of the project because if customer requirements are continuously changing. So there is no deadline. There is a spiral goes to infinite, so in this way spiral model have used. Now the advantages of spiral model, the first important advantage is they identify risk and solve the particular risk in particular project. So this risk parts are first developed and then actual development have started. This is the most important advantage. Spiral model generally used in large and mission critical projects here after the complete development customer feedback have taken. So customer interaction is also most important advantage and customer requirements are continuously changing not a fixed requirements is there and they also create a prototype for creating the prototype. It solve all the errors in prototype. Clear? So these are the advantages. Now the last one is disadvantages of spiral model. The most important disadvantages is here for risk analysis purpose, it required highly particular expertise. That's only risk analysis and risk problems have solved. It is a very costly model that's where doesn't work for smaller projects and spiral model sometimes goes to complex mode because there is a continuously changing requirements and spiral may go to infinite mode that's why and there is a large number of spiral stages means there is a complex documentation have created. So these are the disadvantages of spiral model. So this is all about spiral model. Thank you for your help. In previous session we discussed about software development life cycle with real life examples and generic process model. I have mentioned complete software engineering subject playlist link in below description box. Now in today's session we will discuss about the next important and first model that is waterfall model. Let's start the session. In today's session we will discuss about introduction then when to use waterfall model their phases and advantages and disadvantages of waterfall model. Let's see all these points when by when next. Now the first thing is what exactly waterfall model see waterfall model introduced by Winston Royce in 1970. Waterfall model also called as linear sequential development model in your exam they will ask I'd are explain waterfall model or explain linear sequential development model. The answer is same. See here this is a structure of waterfall model waterfall model is a first software development life cycle model which is widely used in different software engineering projects as per the customer requirements. Now in this waterfall model there are total six phases like requirement analysis and planning than design development testing deployment and maintenance. This model is called as waterfall because see here their diagrammatic representation every cascading components show the waterfalls that's why this is called as waterfall model. Now the main aim of waterfall model is here in this model every phase is completed before the next phase can begin. Means first requirement analysis phase this phase completed then they send output to the design phase. Now design phase take a input of this and this phase send output to the development phase. Now development phase completed and send output to the testing phase means this waterfall model work only in forward direction not a backward direction. This is the main concept of waterfall model next now when to use waterfall model as we know in software engineering there are lots of and different models every model having their own requirements. So waterfall model mainly used in when customer requirement is fixed not change only that way this waterfall model will work. Now waterfall model is not used in complicated and big project it is used in short project and simple project in waterfall model all the tools and technology used in consistent means that is predefined you can't change these tools and technology in ongoing performance of waterfall model. Now waterfall model also having all the resources are well prepared and easily available. So only that way mainly requirements are fixed and used for short and simple project will be prepared next. Now these are the some phases of waterfall model the first phases requirement analysis every software project always start with the communication the communication between the customer and development team or communication between customers and stakeholders. Stakeholder means each and every person involved in particular software project whether it is customer client, tester, project manager, then developer so each and every person involved in software project which is called as stakeholders. So in requirement analysis there is a communication between them so they discuss what exactly customer want and what is the need of the customer means they gather all the requirements from the customer which includes functional requirements means suppose there is a online shopping application so functional requirements means what is the models will be included in particular software what will be the features and specification about software then performance misaccuracy and portability about software and interfacing means their graphical user front end interface. So these all requirements have collected in this particular phase after collecting requirements these requirements maintain in particular document this document are called as SRS that is software requirement specification this document having detailed description of what exactly customer want and requirements of the customer basically this SRS document is a contract between development team and the customer. Now next so after requirement analysis there is a design phase this SRS document send forward to the design phase design phase handled by the UI UX designer team so all the gather requirements are converted into the suitable design so this design are divided into the two parts like high level design and low level design so high level design include complete software architecture means how your project or how your products looks like in future so which includes algorithm like step by solution for example there is a sign up and login page so when you click on login which page will be open so this all design can be drawn in this design phase. So which includes algorithm flow charts then decision tree database design which kind of database tables have included then low level design includes user interface components where is a text box check box have present then rough paper design so all these things have design in this design phase and documented into the software design document that is SDD now this previous phase SRS and SDD send forward to the next phase in this phase also the discuss and finalize which programming languages have used in this particular software project which database and other hardware and software requirements have used this all things have discussed in this particular phase let's see here in this diagram this is a flow chart where is a start where is the end of the project this is a GUI where is a button text box have present and this is a complete software architecture in this way they draw the design by using particular software next now the next one is a development phase so after designing development phases there so development phase means your software design is directly converted into the source code or programming languages so here developer decide the programming languages and database and the first develop a module by software for example in your project in your software there is a login module sign up module then a shopping cart module so they develop a complete project module by and after developing they perform the unit testing here unit testing means one by one they check the module whether it is properly developed or not so complete programming part done here next now next why is the testing phase so in testing phase tester perform all the testing activities and check that whether the particular software meet the customer requirements or not tester mainly perform integration testing here the test complete software at the same time whether it is particularly work or not if there are any errors the report those error and maintain test cases and test reports next now the last phase is deployment and maintenance phase here after completing complete software product this software product is released in market or delivered to the customer this is called as deployment and after delivering if there are any issue have occurred so this thing is called as maintenance so there are some support managers who fix those issues and enhance the product version if customer want any new features they enhance the customer product so these all things have included in maintenance phase now what was the advantages of waterfall model see we already discussed all these phases in detail previously so what exactly advantages of waterfall model first it is very simple and easy to understand the next thing is each phase must be completed see waterfall model going to only forward direction not a backward direction first unit to perform requirement analysis then design then implementation in this way they can't go backward and again change the requirement here requirements are fixed so it works very well in small project and requirements are fixed and understood right so here all the process and results all the things have documented properly clear next now main thing is what was the disadvantages of waterfall model in disadvantages of disadvantages of waterfall model here requirements are fixed this is the main disadvantages because if customer wants some new features so they can't edit those feature in waterfall model that's where there is a high amount of risk and uncertainty next thing is if errors can be occur suppose in requirement analysis errors can be occur that will be solved in the particular phase only you can't go in backward direction suppose in coding phase you you want to change design so you can't go backward in design phase right so this is a main disadvantages this is not a flexible model okay and also the next thing is it is not a good model for complex and long going project and client valuable feedback cannot be included here after completing complete software product only after that client well feedback will be taken basically client feedback taken on each and every phase so this thing is not included in waterfall model clear so by solving all these disadvantages the next spiral model iterative models having there so we will discuss this model in next session so this is all about waterfall model keep learning thank you"
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "textbook_rag"
OLLAMA_MODEL = "mistral:latest"   # change to any model you have pulled, e.g. "mistral"
TOP_K = 5                 # number of chunks to retrieve

SYSTEM_PROMPT = """\
You are a helpful teaching assistant. Your role is to summarize and explain \
textbook content so students can understand and revise topics taught in class.

Use ONLY the context provided below from the textbooks. \
If the context does not contain enough information, say so clearly — do not make things up.

Provide a clear, well-structured answer with headings and bullet points where appropriate.

Context from textbooks:
{context}
"""


# ── Load Vector Store ─────────────────────────────────────────────────────────

def get_vector_store() -> Chroma:
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector database not found at '{CHROMA_DB_DIR}'.\n"
            "Run 'python create_vectordb.py' first to build it."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ── Search & Display ──────────────────────────────────────────────────────────

def run(query: str) -> None:
    print(f"\n🔍 Query: {query}\n")

    db = get_vector_store()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    # ── Section 1: Raw textbook content ──────────────────────────────────────
    print("━" * 70)
    print("📖  TEXTBOOK CONTENT")
    print("━" * 70)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n[{i}] {source}  |  Page {page}")
        print("─" * 70)
        print(doc.page_content.strip())
    print()

    # ── Section 2: LLM summary ───────────────────────────────────────────────
    print("━" * 70)
    print(f"🤖  AI SUMMARY  ({OLLAMA_MODEL})")
    print("━" * 70)
    print("⏳ Generating summary...\n")

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Topic / Question: {question}\n\nSummarize this topic based on the textbook content above."),
    ])

    def format_docs(doc_list):
        parts = []
        for i, doc in enumerate(doc_list, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    print(answer)
    print("━" * 70)



# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(SEARCH_QUERY)
