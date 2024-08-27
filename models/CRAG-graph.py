import os
from langchain_community.chat_models import QianfanChatEndpoint, AzureChatOpenAI
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain import hub
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI
# 需c£È langgraph
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
import pprint

def retrieve(state): #输入{问题} 进行检索 返回字典{文段，问题}
    """
    Retrieve documents
    Args:
    state (dict): The current graph state 状态字典
    Returns:
    state (dict): New key added to state, documents, that contains
    retrieved documents 返回字典{检索文段和问题}
    """
    print("---RETRIEVE---")
    state_dict = state["keys"] #读取字典中的keys
    question = state_dict["question"] #读取keys中的question
    documents = retriever.get_relevant_documents(question) #检索并返回文段
    return {"keys": {"documents": documents, "question": question}}

def generate(state): #输入{文段，问题}，进行回答，输出{文段，问题，回复}
    """
    Generate answer
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM
    generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    llm = QianfanChatEndpoint(streaming=True)
    #llm = QianfanChatEndpoint(model="ERNIE-Bot-4")
    # chat = AzureChatOpenAI(
    # openai_api_version="2023-05-15",
    # azure_deployment=os.getenv('DEPLOYMENT_NAME_GPT3_4K_JP'),
    # temperature=0, # <------ 温t
    # )

    # Post-processing检索文段的后处理
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
    "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state): #输入{问题，文段}，判断每条检索文段的相关性，返回{问题，相关文段，是否网络查询}
    """
    Determines whether the retrieved documents are relevant to the question.确定检索的文档是否与问题相关。
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with relevant documents
    """
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Data model就是创建一个类，用field函数实现里面的score的值只能是yes或no，不然就报错
    class grade(BaseModel):
        """Binary score for relevance check.相关性检索"""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM 原文⽤GPT4 来评分
    #model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = QianfanChatEndpoint(streaming=True)
    # model = AzureChatOpenAI(
    # openai_api_version="2023-05-15",
    # azure_deployment=os.getenv('DEPLOYMENT_NAME_GPT3_4K_JP'),
    # temperature=0, # <------ 温t
    # )

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the
user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the
document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool | parser_tool
    # Score
    filtered_docs = []
    search = "No" # Default do not opt for web search to supplement retrieval不需要网络检索
    for d in documents: #对每项文段进行判断
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)#筛选出有相关性的文段
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes" # Perform web search设置参数，只要一个不相干，就需要网络查询
            continue
    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
        }
    }

def transform_query(state): #输入{问题，文段}，进行问题重述，返回{更好的问题，文段}
    """
    Transform the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for
retrieval. \n 
        Look at the input and try to reason about the underlying sematic
intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    #model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview",streaming=True)
    model = QianfanChatEndpoint(streaming=True)
    #model = QianfanChatEndpoint(model="ERNIE-Bot-4")
    # model = AzureChatOpenAI(
    #     openai_api_version="2023-05-15",
    #     azure_deployment=os.getenv('DEPLOYMENT_NAME_GPT3_4K_JP'),
    #     temperature=0, # <------ 温t
    # )

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    return {"keys": {"documents": documents, "question": better_question}}

def web_search(state): #输入{问题，文段}，进行检索，返回{问题，检索后扩充的文段}
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
    state (dict): The current graph state
    Returns:
    state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    # tool = TavilySearchResults()
    # docs = tool.invoke({"query": question})
    # web_results = "\n".join([d["content"] for d in docs])
    # web_results = Document(page_content=web_results)
    # documents.append(web_results)
    return {"keys": {"documents": documents, "question": question}}

def decide_to_generate(state):#输入{问题，文段，是否网络搜索}，是：返回"transform_query"，否，返回”generate“
    """
    Determines whether to generate an answer or re-generate a question for web
search.决定生成答案还是重新搜索
    Args:
        state (dict): The current state of the agent, including all keys.
    Returns:
        str: Next node to call
    """
    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]
    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]

if __name__ == '__main__':
    # 嵌入模型和数据库设置
    # 使⽤千帆 embedding bge_large_en 模块


    # vectorstore = ElasticsearchStore(
    #     es_url=os.environ['ELASTIC_HOST_HTTP'],
    #     index_name="index_sd_1024_vectors",
    #     embedding=embeddings_model,
    #     es_user="elastic",
    #     vector_query_field='question_vectors',
    #     es_password=os.environ['ELASTIC_ACCESS_PASSWORD']
    # )

    embeddings_model = QianfanEmbeddingsEndpoint(model="bge_large_en", endpoint="bge_large_en")
    index = "../chromadata/save_index"
    vectorstore = Chroma(persist_directory=index, embedding_function=embeddings_model)

    retriever = vectorstore.as_retriever(search_type = 'similarity', search_kwargs={"k":5})


    # 建立langchain Graph
    workflow = StateGraph(GraphState)

    # Define the nodes定义节点
    workflow.add_node("retrieve", retrieve) # retrieve检索
    workflow.add_node("grade_documents", grade_documents) # grade documents是否联网查询
    workflow.add_node("generate", generate) # generate生成
    workflow.add_node("transform_query", transform_query) # transform_query问题重述
    workflow.add_node("web_search", web_search) # web search网络搜查

    # Build graph
    workflow.set_entry_point("retrieve") #先检索kflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("retrieve","grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()

    # 运⾏
    inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])
