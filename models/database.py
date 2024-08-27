from langchain import text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def save_documents_each_chroma(documents, index="save_index", slice_size=100):
    index = "../chromadata/"+index
    print("documents:", str(len(documents)))
    docs = text_splitter.split_documents(documents)
    print("docs:", str(len(docs)))
    len_num = 0
    tmp_num = 0
    tmp_list = []
    for doc in docs:
        len_num += 1
        tmp_num += 1
        tmp_list.append(doc)
        #每隔slice_size存储并展示
        if tmp_num >= slice_size:
            vectordb = Chroma.from_documents(documents=tmp_list, embedding=embedding_model, persist_directory=index)
            tmp_list = []
            tmp_num = 0
        print(f'当前第:{str(len_num)}个chunk')
    ##没到slice_size的最后一组，也存进去
    if tmp_list:
        vectordb = Chroma.from_documents(documents=tmp_list,embedding=embedding_model,persist_directory=index)
    vectordb.persist()
    vectordb = None
    return vectordb

if __name__ == '__main__':

    path = "../knowledge"
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs, show_progress=True, use_multithreading=True)
    docs = loader.load()

    #定义切片splitter和embedding方法
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
          chunk_size=250, chunk_overlap=25
    )
    #embedding_model = OpenAIEmbeddings()
    embedding_model = QianfanEmbeddingsEndpoint(model="bge_large_zh",
                                                 endpoint="bge_large_zh")
    vectordb = save_documents_each_chroma(docs)
    #建立数据库 运⾏⼀次即可 end