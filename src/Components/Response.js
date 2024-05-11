import React from "react";
import styled from "styled-components";
import { FaBook } from "react-icons/fa"; 
import { FaBalanceScale } from "react-icons/fa";
import { FaGavel } from "react-icons/fa";
import { useState } from 'react';

const CaseContainer = styled.div`
  background-color: #cebfb6;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
`;

const LawsContainer = styled.div`
  background-color: #faebd7;
  padding: 10px;
  border-radius: 5px;
  margin-top: 10px;
  margin-bottom: 10px;
`;

const JudgmentContainer = styled.div`
  background-color: #d5d5d5;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
`;


const CommonContainer = styled.div`
  background-color: #222222;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
`;

const HeadingWithIcon = ({ icon, children }) => (
  <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
    {icon}
    <div style={{ marginLeft: "5px" }}>{children}</div>
  </div>
);
const Response = ({ msg }) => {
  const [expandedCase1, setExpandedCase1] = useState(false);
  const [expandedCase2, setExpandedCase2] = useState(false);
  const [moreDetails, setMoreDetails] = useState('');

  
  // Regular expressions to match different sections
  console.log(msg)
  const languageRegex = /^(Tamil|Hindi|English)\s*-\s*(.+)/;

  const hyphenIndex = msg.indexOf('-');
const extractedMsg = hyphenIndex !== -1 ? msg.substring(0, hyphenIndex).trim() : msg.trim();

console.log(extractedMsg);
const toggleExpandCase = (caseDetails) => {
  // Toggle the expanded state
  setExpandedCase1(!expandedCase1);

  // Send a request to the backend to fetch additional details for the specified case
  fetch('/api/more-details', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ 
      caseDetails: caseDetails.substring(0, caseDetails.indexOf("More Details:")) ,
      language: extractedMsg
    })
  })
    .then(response => response.json())
    .then(data => {
      // Update the state with the received more details
      setMoreDetails(data.moreDetails);
    })
    .catch(error => {
      console.error('Error fetching more details:', error);
    });
};

const toggleExpandCase1 = () => {
  setExpandedCase1(!expandedCase1);
};

const toggleExpandCase2 = () => {
  setExpandedCase2(!expandedCase2);
};
  // Extract different sections from the message
  const languageMatch = msg.match(languageRegex);
  console.log(languageMatch)
  if(!(extractedMsg=="English") && !(extractedMsg=="தமிழ்") && !(extractedMsg=="हिंदी")){
    console.log("hmmmm")
    return   <div className="p-3 ">
    <div className="h-fit rounded-lg mt-2 text-sm text-white rounded-bl-none bg-[#222222] p-3 w-fit ml-0">
      <div dangerouslySetInnerHTML={{ __html: msg }} />
    </div>
  </div>
  }
if(extractedMsg=="English"){
const caseDetailsRegex = /Case Details : (.+?)(?=--)/s;
const case1Regex = /Case 1 : (.+?)Case 2 :/s;
const case2Regex = /Case 2 : ([\s\S]+?)(?=\-\-)/s;
const lawsRegex = /--(.+?)(?=Insight:)/gs;
// const judgmentRegex = /JUDGEMENT: (.+)/s;
const judgmentRegex =/Insight: (.*?)(?=\s*Positive:)/s;
const positiveRegex = /Positive:\s*([\d.]+)/s;
const negativeRegex = /Negative:\s*([\d.]+)/s;

  const caseDetailsMatch = msg.match(caseDetailsRegex);
  const case1Match = msg.match(case1Regex);
  const case2Match = msg.match(case2Regex);
  const lawsMatch = msg.match(lawsRegex);
  const posmatch = msg.match(positiveRegex);
  const negmatch=  msg.match(negativeRegex);
  const judgmentMatch = msg.match(judgmentRegex);
  console.log(posmatch)
  const positivePercentage = parseInt(posmatch[1]);
  const negativePercentage = parseInt(negmatch[1]);

  console.log(caseDetailsMatch)
  console.log(case1Match)
  console.log(case2Match)
  console.log(lawsMatch)
  console.log(judgmentMatch)
  if (!caseDetailsMatch || !case1Match || !case2Match || !lawsMatch || !judgmentMatch) {
    return   <div className="p-3 ">
    <div className="h-fit rounded-lg mt-2 text-sm text-white rounded-bl-none bg-[#222222] p-3 w-fit ml-0">
      <div dangerouslySetInnerHTML={{ __html: msg }} />
    </div>
  </div>
  }

  const caseDetails = caseDetailsMatch[1].split("<br>").join("\n");
  const case1 = case1Match[1].split("<br>").join("\n");
  const case2 = case2Match[1].split("<br>").join("\n");
  const colors = ["blue", "green", "red"]; // Define an array of colors

  const laws = lawsMatch[0]
  .split(/--|- -/)
  .filter((law) => law.trim() !== "")
  .map((law, index) => (
    <div key={index} style={{ color: colors[index % colors.length] }}>
      {law}
    </div>
  ));
  const judgment = judgmentMatch[1].split("<br>").join("\n");

  return (
    <div className="p-3 ">
      {/* <CaseContainer>
      <HeadingWithIcon icon={<FaBook />} children="Case Details:" />
        <div style={{ color: 'blue' }}><strong>Case 1:</strong> {case1}</div>
        <div style={{ color: 'green' }}><strong>Case 2:</strong> {case2}</div>
      </CaseContainer> */}
      <CaseContainer>
        <HeadingWithIcon icon={<FaBook />} children="Case Details:" />
        <div style={{ color: 'blue' }}>
          <strong>Case 1:</strong> {case1.includes("More Details:") ? case1.substring(0, case1.indexOf("More Details:")) : case1}

          {case1.includes("More Details:") && !expandedCase1 && (
                <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' More details'}
                </span>
              )}
              {expandedCase1 && (
                <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' Less details'}
                </span>
              )}
              
        </div>
         {expandedCase1 && case1.includes("More Details:") && (
          <div  style={{color:'blue'}}>
            {case1.substring(case1.indexOf("More Details:") + "More Details:".length)}
          </div>
        )} 
        {expandedCase1 && moreDetails && (
        <div style={{ color: 'blue' }}>
          {moreDetails}
        </div>
      )}
        <div style={{color:'green'}}>
        <strong>Case 2:</strong> {case2.includes("More Details:") ? case2.substring(0, case2.indexOf("More Details:")) : case2}

              {case2.includes("More Details:") && !expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' More details'}
                </span>
              )}
              {expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' Less details'}
                </span>
              )}
        </div>
        
        {expandedCase2 && case2.includes("More Details:") && (
          <div  style={{color:'green'}}>
            {case2.substring(case2.indexOf("More Details:") + "More Details:".length)}
          </div>
        )}
      </CaseContainer>
      <LawsContainer>
      <HeadingWithIcon icon={<FaBalanceScale/>} children="Law Details:" />
        <div>{laws}</div>
      </LawsContainer>
      <JudgmentContainer>
        <HeadingWithIcon icon={<FaGavel/>} children="Insight:" />
        <div>{judgment}</div>
      </JudgmentContainer>
      <div className="p-3">
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Positive percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "green", height: "20px", width: `${positivePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{positivePercentage}%</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Negative percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "red", height: "20px", width: `${negativePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{negativePercentage}%</div>
      </div>
    </div>
    </div>
  );
}
else if(extractedMsg=="தமிழ்"){
  // const caseDetailsRegex = /வழக்கு விவரங்கள்: (.+?)(?=\s--)/s;
  const case1Regex = /வழக்கு 1 ?: (.+?)(?=வழக்கு 2 :|$)/s;
  const case2Regex = /வழக்கு 2 : (.+?)(?=--|$)/s;
  const lawsRegex = /--(.+?)(?=நுண்ணறிவு:)/gs;
  // const judgmentRegex = /தீர்ப்பு: (.+)/s;
  const judgmentRegex = /நுண்ணறிவு: (.*?)(?=\s*(?:நேர்மறை|பாசிட்டிவ்):)/s;
  const positiveRegex = /(?:நேர்மறை|பாசிட்டிவ்):\s*([\d.]+)/s;
  const negativeRegex = /(?:எதிர்மறை|பாதகம்):\s*([\d.]+)/s;
// const caseDetailsMatch = msg.match(caseDetailsRegex);
const case1Match = msg.match(case1Regex);
const case2Match = msg.match(case2Regex);
const lawsMatch = msg.match(lawsRegex);
const judgmentMatch = msg.match(judgmentRegex);

const posmatch = msg.match(positiveRegex);
  const negmatch=  msg.match(negativeRegex);
  console.log(posmatch)
  const positivePercentage = parseInt(posmatch[1]);
  const negativePercentage = parseInt(negmatch[1]);

// console.log(caseDetailsMatch);
console.log(case1Match);
console.log(case2Match);
console.log(lawsMatch);
console.log(judgmentMatch);
  if (!case1Match || !case2Match || !lawsMatch || !judgmentMatch) {
    return   <div className="p-3 ">
    <div className="h-fit rounded-lg mt-2 text-sm text-white rounded-bl-none bg-[#222222] p-3 w-fit ml-0">
      <div dangerouslySetInnerHTML={{ __html: msg }} />
    </div>
  </div>
  }

  // const caseDetails = caseDetailsMatch[1].split("<br>").join("\n");
  const case1 = case1Match[1].split("<br>").join("\n");
  const case2 = case2Match[1].split("<br>").join("\n");
  const colors = ["blue", "green", "red"]; // Define an array of colors

  // const laws = lawsMatch.map((law, index) => (
  //   <div key={index} style={{ color: colors[index % colors.length] }}>
  //     {law.trim().slice(2, -5)} {/* Remove leading '--' and trailing '<br>' */}
  //   </div>
  // ));
  const laws = lawsMatch[0]
  .split(/--|- -/)
  .filter(law => law.trim() !== "") // Remove empty strings
  .map((law, index) => (
    <div key={index} style={{ color: colors[index % colors.length] }}>
      {law.trim()} {/* Trim any leading or trailing whitespace */}
    </div>
  ));
  const judgment = judgmentMatch[1].split("<br>").join("\n");

  return (
    <div className="p-3 ">
      {/* <CaseContainer>
      <HeadingWithIcon icon={<FaBook />} children="வழக்கு விவரம்:" />
        <div style={{ color: 'blue' }}><strong>வழக்கு 1:</strong> {case1}</div>
        <div style={{ color: 'green' }}><strong>வழக்கு 2:</strong> {case2}</div>
      </CaseContainer> */}
      <CaseContainer>
        <HeadingWithIcon icon={<FaBook />} children="வழக்கு விவரம்:" />
        <div style={{ color: 'blue' }}>
          <strong>Case 1:</strong> {case1.includes("மேலும் விவரங்கள்:") ? case1.substring(0, case1.indexOf("மேலும் விவரங்கள்:")) : case1}

          {case1.includes("மேலும் விவரங்கள்:") && !expandedCase1 && (
            <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
              {' கூடுதல் தகவல்கள்'}
            </span>
          )}
          {expandedCase1 && (
            <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
              {' குறைவான விவரங்கள்'}
            </span>
          )}
              
        </div>
        {expandedCase1 && case1.includes("மேலும் விவரங்கள்:") && (
          <div  style={{color:'blue'}}>
            {case1.substring(case1.indexOf("மேலும் விவரங்கள்:") + "மேலும் விவரங்கள்:".length)}
          </div>
        )}
        <div style={{color:'green'}}>
        <strong>Case 2:</strong>  {case2.includes("மேலும் விவரங்கள்:") ? case2.substring(0, case2.indexOf("மேலும் விவரங்கள்:")) : case2}
              {case2.includes("மேலும் விவரங்கள்:") && !expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' கூடுதல் தகவல்கள்'}
                </span>
              )}
              {expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' குறைவான விவரங்கள்'}
                </span>
              )}
        </div>
        
        {expandedCase2 && case2.includes("மேலும் விவரங்கள்:") && (
          <div  style={{color:'green'}}>
            {case2.substring(case2.indexOf("மேலும் விவரங்கள்:") + "மேலும் விவரங்கள்:".length)}
          </div>
        )}
      </CaseContainer>
      <LawsContainer>
      <HeadingWithIcon icon={<FaBalanceScale/>} children="சட்ட விவரங்கள்:" />
        <div>{laws}</div>
      </LawsContainer>
      <JudgmentContainer>
        <HeadingWithIcon icon={<FaGavel/>} children="நுண்ணறிவு:" />
        <div>{judgment}</div>
      </JudgmentContainer>
      <div className="p-3">
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Positive percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "green", height: "20px", width: `${positivePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{positivePercentage}%</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Negative percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "red", height: "20px", width: `${negativePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{negativePercentage}%</div>
      </div>
    </div>
    </div>
  );
}
else if(extractedMsg=="हिंदी"){
  const caseDetailsRegex = /मामले का विवरण: (.+?)(?=\s--)/s;
  const case1Regex = /केस 1: (.+?)(?=केस 2:)/s;
  const case2Regex = /केस 2:([\s\S]+?)(?=\-\-)/s;
  const lawsRegex = /--(.+?)(?=अंतर्दृष्टि:)/gs;

  // const judgmentRegex = /निर्णय: दिए गए मामले (.+)/s;
  const judgmentRegex = /अंतर्दृष्टि: केस जीतने की (.*?)(?=\s*सकारात्मक:)/s;
  const positiveRegex = /सकारात्मक:\s*([\d.]+)/s;
  const negativeRegex = /नकारात्मक:\s*([\d.]+)/s;
  const caseDetailsMatch = msg.match(caseDetailsRegex);
  const case1Match = msg.match(case1Regex);
  const case2Match = msg.match(case2Regex);
  const lawsMatch = msg.match(lawsRegex);
  const judgmentMatch = msg.match(judgmentRegex);

  const posmatch = msg.match(positiveRegex);
  const negmatch=  msg.match(negativeRegex);
  console.log(posmatch)
  const positivePercentage = parseInt(posmatch[1]);
  const negativePercentage = parseInt(negmatch[1]);
  
  console.log(caseDetailsMatch)
  console.log(case1Match)
  console.log(case2Match)
  console.log(lawsMatch)
  console.log(judgmentMatch)
  if (!case1Match || !case2Match || !lawsMatch || !judgmentMatch) {
    return   <div className="p-3 ">
    <div className="h-fit rounded-lg mt-2 text-sm text-white rounded-bl-none bg-[#222222] p-3 w-fit ml-0">
      <div dangerouslySetInnerHTML={{ __html: msg }} />
    </div>
  </div>
  }

  // const caseDetails = caseDetailsMatch[1].split("<br>").join("\n");
  const case1 = case1Match[1].split("<br>").join("\n");
  const case2 = case2Match[1].split("<br>").join("\n");
  const colors = ["blue", "green", "red"]; // Define an array of colors

  
  const laws = lawsMatch.map((law, index) => (
    <div key={index} style={{ color: colors[index % colors.length] }}>
      {law.trim().slice(2, -5)} {/* Remove leading '--' and trailing '<br>' */}
    </div>
  ));
  const judgment = "निर्णय: दिए गए मामले "+judgmentMatch[1].split("<br>").join("\n");

  return (
    <div className="p-3 ">
      {/* <CaseContainer>
      <HeadingWithIcon icon={<FaBook />} children="मामले का विवरण:" />
        <div style={{ color: 'blue' }}><strong>मामला 1:</strong> {case1}</div>
        <div style={{ color: 'green' }}><strong>मामला 2:</strong> {case2}</div>
      </CaseContainer> */}
      <CaseContainer>
        <HeadingWithIcon icon={<FaBook />} children="मामले का विवरण:" />
        <div style={{ color: 'blue' }}>
          <strong>Case 1:</strong> {case1.includes("अधिक विवरण:") ? case1.substring(0, case1.indexOf("अधिक विवरण:")) : case1}
          {case1.includes("अधिक विवरण:") && !expandedCase1 && (
            <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
              {' अधिक विवरण'}
            </span>
          )}
          {expandedCase1 && (
            <span onClick={toggleExpandCase1} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
              {' कम विवरण'}
            </span>
          )}
              
        </div>
        <div style={{color:'green'}}>
        <strong>Case 2:</strong> {case2.includes("अधिक विवरण:") ? case2.substring(0, case2.indexOf("अधिक विवरण:")) : case2}
              {case2.includes("अधिक विवरण:") && !expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' अधिक विवरण'}
                </span>
              )}
              {expandedCase2 && (
                <span onClick={toggleExpandCase2} style={{ cursor: "pointer", color: "red", textDecoration: "underline" }}>
                  {' कम विवरण'}
                </span>
              )}
        </div>
        {expandedCase1 && case1.includes("अधिक विवरण:") && (
          <div  style={{color:'blue'}}>
            {case1.substring(case1.indexOf("अधिक विवरण:") + "अधिक विवरण:".length)}
          </div>
        )}
        {expandedCase2 && case2.includes("अधिक विवरण:") && (
          <div  style={{color:'green'}}>
            {case2.substring(case2.indexOf("अधिक विवरण:") + "अधिक विवरण:".length)}
          </div>
        )}
      </CaseContainer>
      <LawsContainer>
      <HeadingWithIcon icon={<FaBalanceScale/>} children="कानून विवरण:" />
        <div>{laws}</div>
      </LawsContainer>
      <JudgmentContainer>
        <HeadingWithIcon icon={<FaGavel/>} children="अंतर्दृष्टि:" />
        <div>{judgment}</div>
      </JudgmentContainer>
      <div className="p-3">
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Positive percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "green", height: "20px", width: `${positivePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{positivePercentage}%</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
        <div style={{ width: "100px" }}>Negative percentage:</div>
        <div style={{ flex: 1, marginLeft: "10px" }}>
          <div style={{ backgroundColor: "red", height: "20px", width: `${negativePercentage}%` }} />
        </div>
        <div style={{ marginLeft: "10px" }}>{negativePercentage}%</div>
      </div>
    </div>
    </div>
  );
}
};
export default Response;
