import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
import pandas as pd
from itertools import combinations
import torch.nn.functional as F
from xaif_eval import xaif
import json
from src.custom_model import MicroAndMacroContextPT2ForTokenClassification
from src.data_utils import prepare_inputs
from src.sttructure_output import ArgumentStructureGenerator
import logging
logging.basicConfig(level=logging.INFO)


class CAASRArgumentStructure:
    def __init__(self, file_obj, model_name="debela-arg/argument_relation_identifier" ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = MicroAndMacroContextPT2ForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


        self.file_obj = file_obj
        self.f_name = file_obj.filename
        self.file_obj.save(self.f_name)
        file = open(self.f_name,'r')

    def is_valid_json(self):
        ''' check if the file is valid json
		'''

        try:
            json.loads(open(self.f_name).read())
        except ValueError as e:			
            return False

        return True
    def is_valid_json_aif(sel,aif_nodes):
        if 'nodes' in aif_nodes and 'locutions' in aif_nodes and 'edges' in aif_nodes:
            return True
        return False
        

    def get_aif(self, format='xAIF'):
        if self.is_valid_json():
            with open(self.f_name) as file:
                data = file.read()
                x_aif = json.loads(data)
                if format == "xAIF":
                    return x_aif
                else:
                    aif = x_aif.get('AIF')
                    return json.dumps(aif)
        else:
            return "Invalid json"

    def load_data(self, nodes):
        arguments = []
        node_id_prpos = {}
        argument =   " '[SEP]' ".join([node.get("text") for node in nodes if node.get("type")=="I"])
        list_prpos = [node.get("nodeID") for node in nodes if node.get("type")=="I"]
        for node in nodes: 
            if node.get("type")=="I":       
                node_id_prpos[node.get("nodeID")] =  node.get("text")    

        combined_texts = []
        for id1, id2 in combinations(list_prpos, 2):
            combined_texts.append((id1, id2))
        proposition_1, proposition_2 = [], []
        for p1, p2 in combined_texts:
            proposition_1.append(node_id_prpos[p1])
            proposition_2.append(node_id_prpos[p2])
            arguments.append(argument)

        rl_label, start_end_indeces = [None] * len(proposition_1), [[]] * len(proposition_1)

        return {"argument": arguments, 'prop_1': proposition_1, 'prop_2': proposition_2, 'label': rl_label, 'start_end_indeces': start_end_indeces},combined_texts,node_id_prpos

    def predict(self, data):
        X_test, pos_embs_test = prepare_inputs(data, self.tokenizer)

        relation_encoder = {0: "None", 1: "Default Inference", 2: "Default Conflict", 3: "Default Rephrase"}
        predictions = []
        confidence = []
        probabilities = []

        for token_id, position_embedding in zip(X_test, pos_embs_test):
            token_id_tensor = torch.tensor(token_id['input_ids']).to(self.device)
            attention_mask = torch.tensor(token_id['attention_mask']).to(self.device)
            tensor_position_embeddings = torch.tensor(position_embedding).to(self.device)

            with torch.no_grad():
                outputs = self.model(token_ids=token_id_tensor, attention_mask=attention_mask,
                                     position_embeddings=tensor_position_embeddings)
                logits = outputs.classification_logits
                predicted_index = logits.argmax(dim=1).item()
                predictions.append(predicted_index)

                logits_softmax = F.softmax(logits, dim=-1)
                confidence.append(logits_softmax[0][predicted_index].item())
                probabilities.append(logits_softmax[0].detach().cpu().numpy())

        return predictions, confidence, probabilities
    
    def get_argument_structure(self,):
        xAIF_input = self.get_aif()
        logging.info(f"xAIF data:  {xAIF_input}, {type(xAIF_input)}, {self.file_obj}")    
        aif = xaif.AIF(xAIF_input)
            
        data, combined_texts, node_id_prpos = self.load_data(aif.aif.get('nodes'))
        predictions, confidence, probabilities = self.predict(data)
        predicted_relations = []
        propositions = []
        for (p1, p2), relation in zip(combined_texts,predictions):
            predicted_relations.append((p1,p2,relation))
            if p1 not in propositions:
                propositions.append(p1)
            if p2 not in propositions:
                propositions.append(p2)

        generator = ArgumentStructureGenerator()
        refined_structure = generator.generate_argument_structure_from_relations(propositions, predicted_relations)
        print(refined_structure)
        relation_encoder = {0: "None", 1: "RA", 2: "CA", 3: "MA"}
        for conclussion_id, premise_relation_list in refined_structure.items():
            print(node_id_prpos[conclussion_id], node_id_prpos[premise_relation_list[0]], relation_encoder[premise_relation_list[1]])
            if relation_encoder[premise_relation_list[1]] in ['CA', 'MA', "RA"]:
                aif.add_component("argument_relation", relation_encoder[premise_relation_list[1]], conclussion_id, premise_relation_list[0])

        return(aif.aif)

if __name__ == "__main__":

    '''
    data = {
    "aif": {
        "nodes": [
            {"nodeID": "1", "text": "THANK YOU", "type": "I", "timestamp": "2016-10-31 17:17:34"},
            {"nodeID": "2", "text": "COOPER : THANK YOU", "type": "L", "timestamp": "2016-11-10 18:34:23"},
            {"nodeID": "3", "text": "You are well come", "type": "I", "timestamp": "2016-10-31 17:17:34"},
            {"nodeID": "4", "text": "Bob : You are well come", "type": "L", "timestamp": "2016-11-10 18:34:23"},
            {"nodeID": "5", "text": "does or doesn't Jeane Freeman think the SNP is divided with what is going on", "type": "I", "timestamp": ""},
            {"nodeID": "6", "text": "the SNP is a big party", "type": "I", "timestamp": ""},
            {"nodeID": "7", "text": "would or wouldn't Jeane Freeman describe the SNP as united", "type": "I", "timestamp": ""},
            {"nodeID": "8", "text": "the SNP has disagreements", "type": "I", "timestamp": ""},
            {"nodeID": "9", "text": "the SNP has disagreements", "type": "I", "timestamp": ""},
            {"nodeID": "10", "text": "Michael Forsyth belongs to a party that has disagreements", "type": "I", "timestamp": ""},
            {"nodeID": "11", "text": "one disagreement of Michael Forsyth's party is currently about their Scottish leader", "type": "I", "timestamp": ""},
            {"nodeID": "12", "text": "Iain Murray has had disagreements with his party", "type": "I", "timestamp": ""},
            {"nodeID": "13", "text": "it's not uncommon for there to be disagreements between party members", "type": "I", "timestamp": ""},
            {"nodeID": "14", "text": "disagreements between party members are entirely to be expected", "type": "I", "timestamp": ""},
            {"nodeID": "15", "text": "what isn't acceptable is any disagreements are conducted that is disrespectful of other points of view", "type": "I", "timestamp": ""},
            {"nodeID": "16", "text": "Jeanne Freeman wants to be in a political party and a country where different viewpoints and different arguments, Donald Dyer famously said, are conducted with respect and without abuse", "type": "I", "timestamp": ""},
            {"nodeID": "17", "text": "who does or doesn't Jeanne Freeman think is being disrespectful then", "type": "I", "timestamp": ""},
            {"nodeID": "18", "text": "people feel, when they have been voicing opinions on different matters, that they have been not listened to", "type": "I", "timestamp": ""},
            {"nodeID": "19", "text": "people feel that they have been treated disrespectfully on all sides of the different arguments and disputes going on", "type": "I", "timestamp": ""}
        ],
        "edges": [
            {"edgeID": "1", "fromID": "247603", "toID": "247602", "formEdgeID": "None"},
            {"edgeID": "2", "fromID": "247604", "toID": "247603", "formEdgeID": "None"}
        ],
        "locutions": [],
        "participants": []
    }
}


strcture_generator  = CAASRArgumentStructure()

structure = strcture_generator.get_argument_structure(data)

print(structure)
'''