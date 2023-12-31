import openai
import constants

openai.api_key = constants.APIKEY
response = openai.ChatCompletion.create(
  model="gpt-4",
  messages = [
        {"role": "system", "content": "You're a real estate expert who explains apartment transactions. give me answer with korean"},
        {"role": "user", "content": "I'd like to ask about the last year's transaction data I have"},
        {"role": "assistant", "content": "Absolutely, I'd be happy to help! Please share the transaction data you have and specify what exactly would you like to know or understand from it."},
        {"role": "user", "content": "구분,2022년 9월,10월,11월,12월,2023년 1월,2월,3월,4월,5월,6월,7월,8월,9월,서울특별시,607,559,727,835,1411,2450,2985,3186,3427,3845,3583,3852,3360, 강남구,31,30,36,40,95,184,181,189,262,252,239,266,194,강동구,19,32,39,46,122,200,178,247,213,229,206,220,180,강북구,12,11,8,45,24,37,77,50,56,126,64,186,49,강서구,30,27,42,50,52,147,146,160,175,206,149,187,190,관악구,17,15,14,14,29,57,62,79,191,96,99,112,108,광진구,11,9,7,57,34,45,48,60,70,64,65,78,78,구로구,41,40,24,26,45,89,129,156,134,134,146,156,144,금천구,14,9,146,13,13,33,52,55,39,57,63,56,54,노원구,30,43,45,57,133,190,188,216,232,272,281,304,257,도봉구,27,17,28,20,67,104,93,86,92,109,118,111,101,동대문구,24,25,31,28,83,106,119,144,155,177,151,139,127,동작구,21,20,15,24,39,66,106,108,120,132,146,138,123,마포구,20,19,19,33,54,92,126,136,166,166,179,167,140,서대문구,14,22,18,26,47,93,106,117,119,128,153,156,142,서초구,20,17,28,29,46,82,120,151,139,181,187,194,141,성동구,14,18,9,25,41,87,112,132,153,170,200,170,168,성북구,33,40,36,50,97,144,152,168,177,181,177,198,178,송파구,29,45,51,86,148,254,230,279,294,288,265,265,256,양천구,31,16,19,28,56,114,110,149,137,173,161,186,175,영등포구,69,29,35,66,55,103,126,159,187,287,169,185,177,용산구,12,8,14,10,13,23,40,43,44,66,50,57,61,은평구,34,27,31,29,54,95,370,145,124,121,143,156,122,종로구,14,4,8,6,7,19,22,17,30,30,22,37,43,중구,11,14,7,11,16,31,30,45,51,53,62,58,52,중랑구,29,22,17,16,41,55,62,95,67,147,88,70,100"},
        {"role": "assistant", "content": "Sure, I see that you've shared transaction data (presumably apartments sold?) for various districts in Seoul over the period from September 2022 to September 2023. Is there a specific question or type of analysis you are looking for with this data?"},
        {"role": "user", "content": "최근들어 거래량의 변화가 가장 큰 위치는 어디야?"}
    ]
)

print(response.choices[0].message.content)