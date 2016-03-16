import os
from Tkinter import *
from PIL import Image, ImageTk
import json
import pdb

IMGDIR = "H:/Dev/pools/images-split/splits/"
DICTFILE = "dict.json"

labeldict = {}

if(os.path.isfile(DICTFILE)):
    with open(DICTFILE,'r') as inputfile:
        labeldict = json.load(inputfile)

root = Tk()

class Application(Frame):
   
    def updatePhoto(self):
        image = Image.open(IMGDIR + self.imglist[self.currenti])
        image = image.resize((256, 256), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)
        self.panel.configure(image=self.photo)
        self.updateLabels()

    def updateLabels(self):
        self.markedCount.configure(text='Marked: ' + str(len(labeldict)))
        self.currentidlabel.configure(text='id: ' + str(self.currenti))
        if (self.imglist[self.currenti] in labeldict):
            self.currentlabel.configure(text='Label: ' + str(labeldict[self.imglist[self.currenti]]))
        else:
            self.currentlabel.configure(text='No Label')

    def createWidgets(self):       
        self.panel = Label(image = self.photo)        
        self.panel.pack(side = "bottom", fill = "both", expand = "yes")        
        root.bind("<Left>",self.leftKey)
        root.bind("<Right>",self.rightKey)
        root.bind("<a>",self.aKey)
        root.bind("<d>",self.dKey)

        self.QUIT = Button(self) 
        self.QUIT["text"] = "SAVE"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.doExit
        self.QUIT.pack({"side": "left"})

        self.markedCount = Label(self, text='Marked: ' + str(len(labeldict)))
        self.markedCount.pack()

        self.currentlabel = Label(self,text='No Label')
        self.currentlabel.pack()

        self.currentidlabel = Label(self,text='id: ' + str(0))
        self.currentidlabel.pack()

    def rightKey(self,event):
        self.currenti = self.currenti + 1
        self.updatePhoto()

    def leftKey(self,event):
        if(self.currenti > 0):
            self.currenti = self.currenti - 1
            self.updatePhoto()

    def aKey(self,event):
        labeldict[self.imglist[self.currenti]] = 0
        self.currenti = self.currenti + 1
        self.updatePhoto()

    def dKey(self,event):
        labeldict[self.imglist[self.currenti]] = 1
        self.currenti = self.currenti + 1
        self.updatePhoto()        

    def doExit(self):
        if(labeldict):
            with open(DICTFILE,'w') as outfile:
                json.dump(labeldict,outfile)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()        
        self.currenti = len(labeldict)
        self.imglist = os.listdir(IMGDIR)
        image = Image.open(IMGDIR + self.imglist[self.currenti])
        self.photo = ImageTk.PhotoImage(image)
        self.createWidgets()
        self.updateLabels()     


app = Application(master=root)
app.mainloop()
root.destroy()