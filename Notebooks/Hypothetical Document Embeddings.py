# Hypothetical Document Embeddings


# reusing the question and vectorstore as earlier
from langchain.prompts import ChatPromptTemplate

# HyDE document genration prompt
template = "\n".join([
    "Act as a Law professor from egypt. Analyze the provided law question and respond to the subsequent question with thoroughness and clarity.",
    "give the answer in details in arabic only ",


    "### Question",
    "{question}",

    "### Answer:",
])
prompt_hyde = ChatPromptTemplate.from_template(template)

# generating a hypothetical document based on the user input
hypothetical_document = llm.invoke(prompt_hyde.format(question=query))





# retrieving relevant document based on hyde document instead of user query
retriever = chroma_db.as_retriever()
retireved_docs = retriever.invoke(hypothetical_document.content)





# Print the results
for doc in retireved_docs:
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 50)




# Use the stuff chain to generate the answer
answer = stuff_chain(
    {
        "input_documents": retireved_docs,  # Pass the search results directly
        "question": query  # Pass the original query
    },
    return_only_outputs=True,  # Return only the output from the chain
)




# Get the output text from the answer
output_text = answer['output_text']

# Print the formatted answer
print("Generated Answer:")
print("=" * 50)  # Separator line
print(output_text)



### في تحسن كبير في الملفات اللي راجعه عكس الفروزين العادي اللي كنت بتعمله علشان كدا حاول تكمل عليه وتقرأ عنه اكتر 