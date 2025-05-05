import json
from google import genai

"""
    This file essentially uses CoT prompting to standardize Question Answers into a format that can be sent to the downstream agents
"""

class Processor:

    def __init__(self):

        self.cotPrompt = ("""
        Original JSON:\n\n
  {
    "Questions": [
      {
        "Question": "What is net sales from cheese in 2018 and 2019 respectively?",
        "Answer": [
          "11,486",
          "11,459"
        ]
      },
      {
        "Question": "What is net sales from cream and other in 2018 and 2019 respectively?",
        "Answer": [
          "5,276",
          "4,228"
        ]
      },
      {
        "Question": "What is net sales from ProBugs Kefir in 2018 and 2019 respectively?",
        "Answer": [
          "2,795",
          "2,780"
        ]
      },
      {
        "Question": "How many product categories are available?",
        "Answer": "6"
      },
      {
        "Question": "What is the change in the net sales for cheese between 2018 and 2019?",
        "Answer": -27
      },
      {
        "Question": "What is the percentage change in net sales from Frozen Kefir between 2018 and 2019?",
        "Answer": 12.76
      }
    ],
    "Images": [],
    "Table": [
      [
        "",
        "",
        "2019",
        "",
        "2018"
      ],
      [
        "In thousands",
        "$",
        "%",
        "$",
        "%"
      ],
      [
        "Drinkable Kefir other than ProBugs",
        "$ 71,822",
        "77%",
        "$ 78,523",
        "76%"
      ],
      [
        "Cheese",
        "11,459",
        "12%",
        "11,486",
        "11%"
      ],
      [
        "Cream and other",
        "4,228",
        "4%",
        "5,276",
        "5%"
      ],
      [
        "ProBugs Kefir",
        "2,780",
        "3%",
        "2,795",
        "3%"
      ],
      [
        "Other dairy",
        "1,756",
        "2%",
        "3,836",
        "4%"
      ],
      [
        "Frozen Kefir (a)",
        "1,617",
        "2%",
        "1,434",
        "1%"
      ],
      [
        "Net Sales",
        "$ 93,662",
        "100%",
        "$ 103,350",
        "100%"
      ]
    ],
    "Text": "Our product categories are:\nDrinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures).\nEuropean-style soft cheeses, including farmer cheese in resealable cups.\nCream and other, which consists primarily of cream, a byproduct of making our kefir.\nProBugs, a line of kefir products designed for children.\nOther Dairy, which includes Cupped Kefir and Icelandic Skyr, a line of strained kefir and yogurt products in resealable cups.\nFrozen Kefir, available in soft serve and pint-size containers.\nLifeway has determined that it has one reportable segment based on how our chief operating decision maker manages the business and in a manner consistent with the internal reporting provided to the chief operating decision maker. The chief operating decision maker, who is responsible for allocating resources and assessing our performance, has been identified collectively as the Chief Financial Officer, the Chief Operating Officer, the Chief Executive Officer, and Chairperson of the board of directors. Substantially all of our consolidated revenues relate to the sale of cultured dairy products that we produce using the same processes and materials and are sold to consumers through a common network of distributors and retailers in the United States.\nNet sales of products by category were as follows for the years ended December 31:\n(a) Includes Lifeway Kefir Shop sales\nSignificant Customers \u2013 Sales are predominately to companies in the retail food industry located within the United States. Two major customers accounted for approximately 22% and 21% of net sales for the years ended December 31, 2019 and 2018, respectively. Two major customers accounted for approximately 17% of accounts receivable as of December 31, 2019 and 2018. Our ten largest customers as a group accounted for approximately 57% and 59% of net sales for the years ended December 31, 2019 and 2018, respectively.",
    "metadata": {
      "table": {
        "uid": "991d23d7-f32d-4954-8e1d-87ad22470fcf",
        "table": [
          [
            "",
            "",
            "2019",
            "",
            "2018"
          ],
          [
            "In thousands",
            "$",
            "%",
            "$",
            "%"
          ],
          [
            "Drinkable Kefir other than ProBugs",
            "$ 71,822",
            "77%",
            "$ 78,523",
            "76%"
          ],
          [
            "Cheese",
            "11,459",
            "12%",
            "11,486",
            "11%"
          ],
          [
            "Cream and other",
            "4,228",
            "4%",
            "5,276",
            "5%"
          ],
          [
            "ProBugs Kefir",
            "2,780",
            "3%",
            "2,795",
            "3%"
          ],
          [
            "Other dairy",
            "1,756",
            "2%",
            "3,836",
            "4%"
          ],
          [
            "Frozen Kefir (a)",
            "1,617",
            "2%",
            "1,434",
            "1%"
          ],
          [
            "Net Sales",
            "$ 93,662",
            "100%",
            "$ 103,350",
            "100%"
          ]
        ]
      },
      "paragraphs": [
        {
          "uid": "6597634c-716e-4a2d-9a19-22f8563a0b1a",
          "order": 1,
          "text": "Our product categories are:"
        },
        {
          "uid": "a4d3952f-4390-4ab2-b6f3-460d14653c10",
          "order": 2,
          "text": "Drinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures)."
        },
        {
          "uid": "d623137a-e787-4204-952a-af9d4ed3a2db",
          "order": 3,
          "text": "European-style soft cheeses, including farmer cheese in resealable cups."
        },
        {
          "uid": "2fec11a0-66f7-4927-b9c2-8090d7694285",
          "order": 4,
          "text": "Cream and other, which consists primarily of cream, a byproduct of making our kefir."
        },
        {
          "uid": "082528f8-e187-48aa-b7b9-8e0fa77899fb",
          "order": 5,
          "text": "ProBugs, a line of kefir products designed for children."
        },
        {
          "uid": "f6e4a899-1c8d-45ab-b42b-c869f43f0b21",
          "order": 6,
          "text": "Other Dairy, which includes Cupped Kefir and Icelandic Skyr, a line of strained kefir and yogurt products in resealable cups."
        },
        {
          "uid": "2bea93c7-0a63-4134-b886-0e94850bb645",
          "order": 7,
          "text": "Frozen Kefir, available in soft serve and pint-size containers."
        },
        {
          "uid": "616fb222-13b2-4ef9-82b8-7d8c7c19053a",
          "order": 8,
          "text": "Lifeway has determined that it has one reportable segment based on how our chief operating decision maker manages the business and in a manner consistent with the internal reporting provided to the chief operating decision maker. The chief operating decision maker, who is responsible for allocating resources and assessing our performance, has been identified collectively as the Chief Financial Officer, the Chief Operating Officer, the Chief Executive Officer, and Chairperson of the board of directors. Substantially all of our consolidated revenues relate to the sale of cultured dairy products that we produce using the same processes and materials and are sold to consumers through a common network of distributors and retailers in the United States."
        },
        {
          "uid": "b34f5f62-04d9-464c-a3e4-6774cf82434a",
          "order": 9,
          "text": "Net sales of products by category were as follows for the years ended December 31:"
        },
        {
          "uid": "9b4f9b4b-1bdd-408a-a01c-873f012d75a1",
          "order": 10,
          "text": "(a) Includes Lifeway Kefir Shop sales"
        },
        {
          "uid": "a002d755-a4d0-44a0-af65-c5cf4692480e",
          "order": 11,
          "text": "Significant Customers \u2013 Sales are predominately to companies in the retail food industry located within the United States. Two major customers accounted for approximately 22% and 21% of net sales for the years ended December 31, 2019 and 2018, respectively. Two major customers accounted for approximately 17% of accounts receivable as of December 31, 2019 and 2018. Our ten largest customers as a group accounted for approximately 57% and 59% of net sales for the years ended December 31, 2019 and 2018, respectively."
        }
      ],
      "questions": [
        {
          "uid": "9ef99ae1-e17b-4b51-b81e-48cc48a481d1",
          "order": 1,
          "question": "What is net sales from cheese in 2018 and 2019 respectively?",
          "answer": [
            "11,486",
            "11,459"
          ],
          "derivation": "",
          "answer_type": "multi-span",
          "answer_from": "table",
          "rel_paragraphs": [],
          "req_comparison": false,
          "scale": "thousand"
        },
        {
          "uid": "49995e73-6740-40f5-9322-cfa4f0a4377d",
          "order": 2,
          "question": "What is net sales from cream and other in 2018 and 2019 respectively?",
          "answer": [
            "5,276",
            "4,228"
          ],
          "derivation": "",
          "answer_type": "multi-span",
          "answer_from": "table",
          "rel_paragraphs": [],
          "req_comparison": false,
          "scale": "thousand"
        },
        {
          "uid": "0a36372a-151d-432f-becb-5b2d738f0e56",
          "order": 3,
          "question": "What is net sales from ProBugs Kefir in 2018 and 2019 respectively?",
          "answer": [
            "2,795",
            "2,780"
          ],
          "derivation": "",
          "answer_type": "multi-span",
          "answer_from": "table",
          "rel_paragraphs": [],
          "req_comparison": false,
          "scale": "thousand"
        },
        {
          "uid": "2965da5c-4d2a-402c-85c6-c1514f998285",
          "order": 4,
          "question": "How many product categories are available?",
          "answer": "6",
          "derivation": "Drinkable Kefir other than ProBugs ## Cheese ## Cream and other ## ProBugs Kefir ## Other dairy ## Frozen Kefir",
          "answer_type": "count",
          "answer_from": "table-text",
          "rel_paragraphs": [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
          ],
          "req_comparison": false,
          "scale": ""
        },
        {
          "uid": "847400ae-6d72-4afa-9b2c-c9677578034a",
          "order": 5,
          "question": "What is the change in the net sales for cheese between 2018 and 2019?",
          "answer": -27,
          "derivation": " 11,459 - 11,486 ",
          "answer_type": "arithmetic",
          "answer_from": "table",
          "rel_paragraphs": [],
          "req_comparison": false,
          "scale": "thousand"
        },
        {
          "uid": "1c78b8ad-5dd7-471e-bafd-71e1515afaa3",
          "order": 6,
          "question": "What is the percentage change in net sales from Frozen Kefir between 2018 and 2019?",
          "answer": 12.76,
          "derivation": "(1,617-1,434)/1,434",
          "answer_type": "arithmetic",
          "answer_from": "table",
          "rel_paragraphs": [],
          "req_comparison": false,
          "scale": "percent"
        }
      ]
    }
  }\n\n

Modified JSON:\n\n

{
  "Questions": [
    {
      "Question": "What is net sales from cheese in 2018 and 2019 respectively?",
      "Answer": [
        "11,486",
        "11,459"
      ],
      "modality": [
        "table"
      ]
    },
    {
      "Question": "What is net sales from cream and other in 2018 and 2019 respectively?",
      "Answer": [
        "5,276",
        "4,228"
      ],
      "modality": [
        "table"
      ]
    },
    {
      "Question": "What is net sales from ProBugs Kefir in 2018 and 2019 respectively?",
      "Answer": [
        "2,795",
        "2,780"
      ],
      "modality": [
        "table"
      ]
    },
    {
      "Question": "How many product categories are available?",
      "Answer": "6",
      "modality": [
        "text",
        "table"
      ]
    },
    {
      "Question": "What is the change in the net sales for cheese between 2018 and 2019?",
      "Answer": -27,
      "modality": [
        "table"
      ]
    },
    {
      "Question": "What is the percentage change in net sales from Frozen Kefir between 2018 and 2019?",
      "Answer": 12.76,
      "modality": [
        "table"
      ]
    }
  ],
  "Images": [],
  "Table": [
    [
      "",
      "",
      "2019",
      "",
      "2018"
    ],
    [
      "In thousands",
      "$",
      "%",
      "$",
      "%"
    ],
    [
      "Drinkable Kefir other than ProBugs",
      "$ 71,822",
      "77%",
      "$ 78,523",
      "76%"
    ],
    [
      "Cheese",
      "11,459",
      "12%",
      "11,486",
      "11%"
    ],
    [
      "Cream and other",
      "4,228",
      "4%",
      "5,276",
      "5%"
    ],
    [
      "ProBugs Kefir",
      "2,780",
      "3%",
      "2,795",
      "3%"
    ],
    [
      "Other dairy",
      "1,756",
      "2%",
      "3,836",
      "4%"
    ],
    [
      "Frozen Kefir (a)",
      "1,617",
      "2%",
      "1,434",
      "1%"
    ],
    [
      "Net Sales",
      "$ 93,662",
      "100%",
      "$ 103,350",
      "100%"
    ]
  ],
  "Text": "Our product categories are:\nDrinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures).\nEuropean-style soft cheeses, including farmer cheese in resealable cups.\nCream and other, which consists primarily of cream, a byproduct of making our kefir.\nProBugs, a line of kefir products designed for children.\nOther Dairy, which includes Cupped Kefir and Icelandic Skyr, a line of strained kefir and yogurt products in resealable cups.\nFrozen Kefir, available in soft serve and pint-size containers.\nLifeway has determined that it has one reportable segment based on how our chief operating decision maker manages the business and in a manner consistent with the internal reporting provided to the chief operating decision maker. The chief operating decision maker, who is responsible for allocating resources and assessing our performance, has been identified collectively as the Chief Financial Officer, the Chief Operating Officer, the Chief Executive Officer, and Chairperson of the board of directors. Substantially all of our consolidated revenues relate to the sale of cultured dairy products that we produce using the same processes and materials and are sold to consumers through a common network of distributors and retailers in the United States.\nNet sales of products by category were as follows for the years ended December 31:\n(a) Includes Lifeway Kefir Shop sales\nSignificant Customers \u2013 Sales are predominately to companies in the retail food industry located within the United States. Two major customers accounted for approximately 22% and 21% of net sales for the years ended December 31, 2019 and 2018, respectively. Two major customers accounted for approximately 17% of accounts receivable as of December 31, 2019 and 2018. Our ten largest customers as a group accounted for approximately 57% and 59% of net sales for the years ended December 31, 2019 and 2018, respectively.",
  "metadata": {
    "table": {
      "uid": "991d23d7-f32d-4954-8e1d-87ad22470fcf",
      "table": [
        [
          "",
          "",
          "2019",
          "",
          "2018"
        ],
        [
          "In thousands",
          "$",
          "%",
          "$",
          "%"
        ],
        [
          "Drinkable Kefir other than ProBugs",
          "$ 71,822",
          "77%",
          "$ 78,523",
          "76%"
        ],
        [
          "Cheese",
          "11,459",
          "12%",
          "11,486",
          "11%"
        ],
        [
          "Cream and other",
          "4,228",
          "4%",
          "5,276",
          "5%"
        ],
        [
          "ProBugs Kefir",
          "2,780",
          "3%",
          "2,795",
          "3%"
        ],
        [
          "Other dairy",
          "1,756",
          "2%",
          "3,836",
          "4%"
        ],
        [
          "Frozen Kefir (a)",
          "1,617",
          "2%",
          "1,434",
          "1%"
        ],
        [
          "Net Sales",
          "$ 93,662",
          "100%",
          "$ 103,350",
          "100%"
        ]
      ]
    },
    "paragraphs": [
      {
        "uid": "6597634c-716e-4a2d-9a19-22f8563a0b1a",
        "order": 1,
        "text": "Our product categories are:"
      },
      {
        "uid": "a4d3952f-4390-4ab2-b6f3-460d14653c10",
        "order": 2,
        "text": "Drinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures)."
      },
      {
        "uid": "d623137a-e787-4204-952a-af9d4ed3a2db",
        "order": 3,
        "text": "European-style soft cheeses, including farmer cheese in resealable cups."
      },
      {
        "uid": "2fec11a0-66f7-4927-b9c2-8090d7694285",
        "order": 4,
        "text": "Cream and other, which consists primarily of cream, a byproduct of making our kefir."
      },
      {
        "uid": "082528f8-e187-48aa-b7b9-8e0fa77899fb",
        "order": 5,
        "text": "ProBugs, a line of kefir products designed for children."
      },
      {
        "uid": "f6e4a899-1c8d-45ab-b42b-c869f43f0b21",
        "order": 6,
        "text": "Other Dairy, which includes Cupped Kefir and Icelandic Skyr, a line of strained kefir and yogurt products in resealable cups."
      },
      {
        "uid": "2bea93c7-0a63-4134-b886-0e94850bb645",
        "order": 7,
        "text": "Frozen Kefir, available in soft serve and pint-size containers."
      },
      {
        "uid": "616fb222-13b2-4ef9-82b8-7d8c7c19053a",
        "order": 8,
        "text": "Lifeway has determined that it has one reportable segment based on how our chief operating decision maker manages the business and in a manner consistent with the internal reporting provided to the chief operating decision maker. The chief operating decision maker, who is responsible for allocating resources and assessing our performance, has been identified collectively as the Chief Financial Officer, the Chief Operating Officer, the Chief Executive Officer, and Chairperson of the board of directors. Substantially all of our consolidated revenues relate to the sale of cultured dairy products that we produce using the same processes and materials and are sold to consumers through a common network of distributors and retailers in the United States."
      },
      {
        "uid": "b34f5f62-04d9-464c-a3e4-6774cf82434a",
        "order": 9,
        "text": "Net sales of products by category were as follows for the years ended December 31:"
      },
      {
        "uid": "9b4f9b4b-1bdd-408a-a01c-873f012d75a1",
        "order": 10,
        "text": "(a) Includes Lifeway Kefir Shop sales"
      },
      {
        "uid": "a002d755-a4d0-44a0-af65-c5cf4692480e",
        "order": 11,
        "text": "Significant Customers \u2013 Sales are predominately to companies in the retail food industry located within the United States. Two major customers accounted for approximately 22% and 21% of net sales for the years ended December 31, 2019 and 2018, respectively. Two major customers accounted for approximately 17% of accounts receivable as of December 31, 2019 and 2018. Our ten largest customers as a group accounted for approximately 57% and 59% of net sales for the years ended December 31, 2019 and 2018, respectively."
      }
    ],
    "questions": [
      {
        "uid": "9ef99ae1-e17b-4b51-b81e-48cc48a481d1",
        "order": 1,
        "question": "What is net sales from cheese in 2018 and 2019 respectively?",
        "answer": [
          "11,486",
          "11,459"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "thousand"
      },
      {
        "uid": "49995e73-6740-40f5-9322-cfa4f0a4377d",
        "order": 2,
        "question": "What is net sales from cream and other in 2018 and 2019 respectively?",
        "answer": [
          "5,276",
          "4,228"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "thousand"
      },
      {
        "uid": "0a36372a-151d-432f-becb-5b2d738f0e56",
        "order": 3,
        "question": "What is net sales from ProBugs Kefir in 2018 and 2019 respectively?",
        "answer": [
          "2,795",
          "2,780"
        ],
        "derivation": "",
        "answer_type": "multi-span",
        "answer_from": "table",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "thousand"
      },
      {
        "uid": "2965da5c-4d2a-402c-85c6-c1514f998285",
        "order": 4,
        "question": "How many product categories are available?",
        "answer": "6",
        "derivation": "Drinkable Kefir other than ProBugs ## Cheese ## Cream and other ## ProBugs Kefir ## Other dairy ## Frozen Kefir",
        "answer_type": "count",
        "answer_from": "table-text",
        "rel_paragraphs": [
          "2",
          "3",
          "4",
          "5",
          "6",
          "7"
        ],
        "req_comparison": false,
        "scale": ""
      },
      {
        "uid": "847400ae-6d72-4afa-9b2c-c9677578034a",
        "order": 5,
        "question": "What is the change in the net sales for cheese between 2018 and 2019?",
        "answer": -27,
        "derivation": " 11,459 - 11,486 ",
        "answer_type": "arithmetic",
        "answer_from": "table",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "thousand"
      },
      {
        "uid": "1c78b8ad-5dd7-471e-bafd-71e1515afaa3",
        "order": 6,
        "question": "What is the percentage change in net sales from Frozen Kefir between 2018 and 2019?",
        "answer": 12.76,
        "derivation": "(1,617-1,434)/1,434",
        "answer_type": "arithmetic",
        "answer_from": "table",
        "rel_paragraphs": [],
        "req_comparison": false,
        "scale": "percent"
      }
    ]
  }
}
        """)
        self.seperator = "\n\n##############################\n\n"
        self.lineBreak = "\n\n"
        self.prompt = ("Inside every question block I would like you to add another field called modality, model it as a list. If you think that we need to use the text to answer the question, please add text, if you think we need the table to answer the question add table. If you think that we need both, then add both. Then return the json back to me.")

    def prepareMessage(self, data):

        return (
            self.cotPrompt
            + self.seperator
            + self.lineBreak
            + json.dumps(data, indent=2)
            + self.lineBreak
            + self.prompt
        )

    def callGmeini(self, msg):

        # this is defunct
        API_KEY = "AIzaSyAX2vBrA8q1WxowKA0f7SUHBEbabAZ4uuE"

        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents="Explain how AI works"
        )
        return response.text

    def addModality(self, data):

        msg = self.prepareMessage(data)
        return self.callGmeini(msg)

    def process_json_file(self, input_file, output_file):
        """
        Reads a file containing a list of JSON objects, processes them to extract
        relevant fields and store them in a file
        """

        with open(file=input_file, mode="r", encoding='utf-8') as f:
            data_list = json.load(fp=f)

        transformed_list = []
        for data in data_list:
            transformed_list.append(self.addModality(data))

        # Write the transformed data to the output JSON file
        with open(file=output_file, mode="w", encoding='utf-8') as f:
            json.dump(transformed_list, fp=f, indent=2)

        print(f"Transformed data written to {output_file}")

if __name__ == "__main__":
    input_file = "../../../data/tatqa_dataset_train_preprocessed.json"
    output_file = "../../../output/tatqa_dataset.json"
    processorObj = Processor()
    processorObj.process_json_file(input_file, output_file)