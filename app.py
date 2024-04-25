import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Product Description Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Product Description Generator🛒")
st.sidebar.markdown("### Welcome to the Lyzr Product Description Generator!")
st.sidebar.markdown("This app harnesses power of Lyzr Automata to Create Product description for your Products. You have to Enter Product Name and Specification and it will Generate SEO Friendly Product Description for Product.")
st.markdown("This app uses Lyzr Automata Agent to Generate Product Description based on your Product Name and Specification.")


open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

name = st.sidebar.text_input("Enter Your Product Name", placeholder="Samsung Galaxy S24")
specification = st.sidebar.text_area("Enter your Product Specification", placeholder="Google",height=200)


def product_description(product_name,specs):

    pd_agent = Agent(
            role="Product Description expert",
            prompt_persona=f"You are an Expert Ecommerce Expert.Your Task Is to write SEO Friendly Product Description."
        )

    prompt = f"""Write a Product Description for below product with attached specification.
    product: {product_name}
    specification: {specs}
    
    Follow Below Instruction:
    1/ Understand who your target customers are and tailor the description to their needs, preferences, and interests.
    2/ Identify the most important features of the product and emphasize them in your description. Focus on what sets the product apart from others.
    3/ Incorporate relevant keywords into your description to improve search engine visibility and attract organic traffic to your product page.
    4/ Write in a clear and straightforward manner, avoiding jargon or technical language that might confuse customers. Keep sentences and paragraphs short for easy reading.
    
    Example Description:
    Split AC with inverter compressor: Variable speed compressor which adjusts power depending on heat load. Powered with 23000 microholes, you can enjoy powerful and gentle cooling. With Convertible 5in1 modes, you can change according to your mood and requirement
    Capacity (1.5Ton): Suitable for medium sized rooms (111 to 150 sq ft)
    Energy Rating: 5 Star BEE Rating with Power Saving Mode | ISEER rating of 5.16 W/W (better than industry benchmarks | Electricity Consumption : 749.48 Units Per Year
    Warranty: 1 Year Standard Warranty on Product, 1 Year Warranty on PCB, 10 Years Warranty on Digital Inverter Compressor
    Copper Condenser Coil: Better cooling and requires low maintenance
    Key Features: Convertible 5in1, 4 way swing, 3 Step Auto Clean, Easy to Clean Filter, Copper Anti-Bacterial Filter, Coated Copper tubes
    Special Features: Windfree Cooling, Windfree Good Sleep, Durafin Ultra, Triple Protection Plus, Digital Inverter Technology
    Refrigerant Gas: R32 - Environmental Friendly - No Ozone Depletion Potential. The air conditioner uses the next generation R32 refrigerant, which helps conserve the ozone layer and has a low impact on global warming
    Indoor Unit Dimensions (mm) WxHxD): 1055 X 299 X 215 mm | Outdoor Unit Dimensions (mm) WxHxD): 880 X 638 X 310 mm | Indoor Weight 11.4 | Outdoor Weight 37
    Included In The Box: 1 Indoor Unit, 1 Outdoor Unit, Copper Pipe (3 meter standard) 1 unit , 1 Remote, 1 Manuals , 2 Batteries
    """

    pd_task = Task(
        name="Generate Product Description",
        model=open_ai_text_completion_model,
        agent=pd_agent,
        instructions=prompt,
    )

    output = LinearSyncPipeline(
        name="Product Description Pipline",
        completion_message="Product Description Generated!!",
        tasks=[
              pd_task
        ],
    ).run()

    answer = output[0]['task_output']

    return answer


if st.sidebar.button("Generate", type="primary"):
    solution = product_description(name, specification)
    st.markdown(solution)

