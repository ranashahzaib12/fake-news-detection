import streamlit as st
import pickle
import pandas as pd

# Load the pickled model, vectorizer, and encoder
with open("random_forest_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Sample articles (title, text)
sample_articles = [
    {
        "title": "Trump remains the favorite to arrive at the convention with the most delegates to his name",
        "text": "Trump remains the favorite to arrive at the convention with the most delegates to his name, but he’s far from assured of the majority of delegates he’d need to win the nomination on the first ballot. Given that reality, the GOP front-runner recently retooled his campaign to address his clear weaknesses in the under-the-radar battle to send loyal delegates to the convention. If those efforts don't start paying dividends soon, though, Trump very well may arrive in Cleveland with the most delegates—and leave without the nomination."
    },
    {
        "title": "Strong Solar Storm, Tech Risks Today | S0 News Oct.26.2016 [VIDEO]",
        "text": "Click Here To Learn More About Alexandra's Personalized Essences Psychic Protection Click Here for More Information on Psychic Protection! Implant Removal Series Click here to listen to the IRP and SA/DNA Process Read The Testimonials Click Here To Read What Others Are Experiencing! Copyright © 2012 by Galactic Connection. All Rights Reserved. Excerpts may be used, provided that full and clear credit is given to Alexandra Meadors and www.galacticconnection.com with appropriate and specific direction to the original content. Unauthorized use and/or duplication of any material on this website without express and written permission from its author and owner is strictly prohibited. Thank you. Privacy Policy By subscribing to GalacticConnection.com you acknowledge that your name and e-mail address will be added to our database. As with all other personal information, only working affiliates of GalacticConnection.com have access to this data. We do not give GalacticConnection.com addresses to outside companies, nor will we ever rent or sell your email address. Any e-mail you send to GalacticConnection.com is completely confidential. Therefore, we will not add your name to our e-mail list without your permission. Continue reading... Galactic Connection 2016 | Design & Development by AA at Superluminal Systems Sign Up for Our Newsletter Join our newsletter to receive exclusive updates, interviews, discounts, and more. Join Us!"
    },
    {
        "title": "10 Ways America Is Preparing for World War 3",
        "text": "October 31, 2016 at 4:52 am Pretty factual except for women in the selective service. American military is still voluntary only and hasn't been a draft since Vietnam war. The comment was made by a 4 star general of the army about drafting women and he said it to shut up liberal yahoos."
    },
    {
        "title": "Trump takes on Cruz, but lightly",
        "text": "Killing Obama administration rules, dismantling Obamacare and pushing through tax reform are on the early to-do list."
    },
    {
    "title": "Hillary Clinton Makes A Bipartisan Appeal on Staten Island",
    "text": "Hillary Clinton told a Staten Island crowd today that she was the candidate who could reach across party lines to get things done as president—pointing to her experience representing the borough in the Senate and even giving public thanks to Republican President George W. Bush."
},
{
    "title": "New Senate majority leader’s main goal for GOP: Don’t be scary",
    "text": "Mitch McConnell has an unusual admonition for the new Republican majority as it takes over the Senate this week: Don’t be 'scary.' The incoming Senate majority leader has set a political goal for the next two years of overseeing a functioning, reasonable majority on Capitol Hill."
},
{
    "title": "Anti-Trump forces seek last-ditch delegate revolt",
    "text": "The faction of the GOP that is unhappy with Donald Trump as the party's presumptive nominee has one last plan to stop the mogul: staging an all-out delegate revolt at the Republican National Convention."
},

{
    "title": "Sanders Trounces Clinton in W. Va. -- But Will It Make a Difference?",
    "text": "Democrat Bernie Sanders picked up more delegates in West Virginia than Hillary Clinton. The Vermont senator's still way behind, but says he's not giving up, calling his win in West Virginia 'tremendous.'"
},{
    "title": "Pure chaos: Donald Trump’s campaign management offers a glimpse into his governing style",
    "text": "If you want a glimpse into a presidential candidate’s governing style, take a look at his campaign. Donald Trump has managed his campaign the way he manages his casinos or his reality TV program: haphazardly and with an unearned arrogance."
},{
    "title": "Donald Trump Is Changing His Campaign Slogan to Prove He’s Not Racist",
    "text": "After a week of nonstop criticism for comments many condemned as racially charged, Donald Trump claims to be altering his campaign to be more inclusive by adding 'for everyone' to his 'Make America Great Again' slogan."
}

]

# Streamlit App UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown(
    "<span style='color:red; font-weight:bold;'>Note:</span> This demo can only predict the article labels for the given 10 sample articles listed below. Custom text prediction is not available in this version.",
    unsafe_allow_html=True
)
st.markdown("Paste a news **title and article** or select a sample to see if it's **FAKE** or **REAL**.")

# Choose sample article
sample_titles = [""] + [f"{i+1}. {article['title']}" for i, article in enumerate(sample_articles)]
sample_choice = st.selectbox("Or choose a sample article:", sample_titles)

# Auto-fill title and text from sample if selected
default_title = ""
default_text = ""
if sample_choice and sample_choice != "":
    idx = int(sample_choice.split('.')[0]) - 1
    default_title = sample_articles[idx]["title"]
    default_text = sample_articles[idx]["text"]

# Inputs for user
title_input = st.text_input("📝 Title", value=default_title)
text_input = st.text_area("📰 Article Text", value=default_text, height=200)

# Predict button
if st.button("🔍 Predict"):
    if not title_input.strip() or not text_input.strip():
        st.warning("Please enter both title and text.")
    else:
        # Combine title and text like in training
        combined = title_input.strip() + " " + text_input.strip()
        transformed = tfidf_vectorizer.transform([combined])
        prediction = model.predict(transformed)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        # Display result
        st.subheader("📢 Prediction:")
        if predicted_label.lower() == "fake":
            st.error("🚫 This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **REAL**.")
