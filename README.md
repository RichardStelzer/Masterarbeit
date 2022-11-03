# Masterarbeit
Dieses Verzeichnis enthält ein paar Sourcecodeausschnitte meiner Masterarbeit (Titel: "Entwicklung einer vollautomatisierten Pipeline zur qualitativ hochwertigen Erfassung von Fahrzeugen").

Eine kurze Zusammenfassung gibt es auf der Institutswebsite &rarr; [Link zu Webpräsentation](https://www.ifp.uni-stuttgart.de/lehre/masterarbeiten/615-Stelzer2/). 

Ergänzend dazu gibt es hier eine weitere grafische Darstellung der Pipeline:

<p align="center"><img src="/assets/pipeline_ueberblick.png" alt="Pipeline Überblick" width="700"/></p>

Die fertige GUI (```gui_main.py```) mit deren Hilfe die einzelnen Pipelineschritte gesteuert werden können ist in der folgenden Abbildung zu sehen. Die Pipeline wird durchlaufen, indem die Schritte 1-13 nacheinander abgearbeitet werden:

<p align="center"><img src="/assets/gui_ergebnis1.PNG" alt="GUI" width="700"/></p>

Das Admin-Webinterface zur Überprüfung von Erfassungen ist im Ordner ```admin_webinterface``` zu finden und basiert auf den Standard Webtechnologien HTML, JS und PHP.

Der Datenaustausch zwischen dem lokalen System, dem Webserver und der Crowdsourcing Plattform Microworkers.com ist im nachfolgenden Schema dargestellt:

<p align="center"><img src="/assets/methodik_ueberblick_daten.png" alt="Datenaustausch zwischen einzelnen Plattformen" width="550"/></p>
