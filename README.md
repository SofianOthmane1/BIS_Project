# Overview 

This repository documents the Multi-Layer Contagion Model I developed at the BIS Innovation Hub in London, hosted by the Bank of England. This project was written in collaboration with the BIS and was in part-fulfillment of my MSc in Data Science and Economics at UCL. 

The code base covers the entire pipeline from taking real-world bank balance sheet data and converting it into a multi-layer node level input data. Then by stochastically simulating 10 000 networks using Cimini et al. (2015) link probability fitness model, representing 12 million bilateral exposures we shock each network throguh 3 scenarios to simulate contagion within the networks, this is performed using the DebtRank algorithm. From these outputs we can then analyse how topology mitigates or catalyses contagion.

A second experiment explores how the training of a GNN could prove benefit to supervisors and researchers in Financial Stability by predicting the contagion pathway with extremely high accuracuy once given the failing bank. In a final test we remove edge data sequentially from the input graphs to see how the model performs under less bilateral data, proving peformance under real-world data accessibility.
