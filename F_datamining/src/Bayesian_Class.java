import java.io.*;
import java.util.*;


public class Bayesian_Class {
	public static void main(String[] args) throws IOException {
			  String[][] dataset = new String [120][8] ; //dataNum 0~119,attribute 0~7
			  String[][] random = new String [80][8] ; //從120筆隨機抽取80筆建立training
			  double[][] D_probability = new double[8][6] ; 
			  double[][] a1_init = new double[4][2] ; //求a1 mean&varience
			  //P(Temperature=x.x|D1=yes),P(Temperature=x.x|D2=yes)
			  //P(Temperature=x.x|D1=no),P(Temperature=x.x|D2=yes)
			  //P(Temperature=x.x|D1=yes),P(Temperature=x.x|D2=no)
			  //P(Temperature=x.x|D1=no),P(Temperature=x.x|D2=no)
			  String[][] datatest = new String[40][8] ; //從120筆資料抽40筆跑test
			  String[][] testDecision = new String[40][2] ;//[][0] = 猜D1,[][1] = 猜D2
			  int attNum = 0 ;
		      int i = 0;
		      Scanner sc = null;
		 
		      sc = new Scanner(new File("D:/DataMining_Final_Project/dataset.txt"));

		      while(sc.hasNext()){ //dataset
		    	  if(i<120){ 
		    		  dataset[i][attNum] = sc.next();
		    		  //System.out.println(i+"//"+attNum+"//"+dataset[i][attNum]); 
		    		  attNum ++ ;
		    		  if(attNum % 8 == 0){
		    			  attNum = 0 ;
		    			  i++ ;
		    		  }
		    	  }
		    	  
		      }
		      sc.close();
		      
		      random = random_80sample(dataset) ;//隨機抽樣80筆
		      D_probability = event_probability(random) ;
		      a1_init = sample_mean_variance(random);
		      
		      datatest = random_40sample(dataset) ;
		      
		      for(int z = 0 ; z < 40 ; z++){ //預測資料
		    	  testDecision[z][0] = D1_Test_model(dubdigit(datatest[z][0]),datatest[z][1],datatest[z][2],datatest[z][3],datatest[z][4],datatest[z][5],D_probability,a1_init) ;
		    	  testDecision[z][1] = D2_Test_model(dubdigit(datatest[z][0]),datatest[z][1],datatest[z][2],datatest[z][3],datatest[z][4],datatest[z][5],D_probability,a1_init) ;
		      }
		      
		      double cor_D1 = 0 , cor_D2 = 0 ;
		      double ans_D1 = 0.0 , ans_D2 = 0.0 ;
		      double ansD1yes = 0.0 , ansD2yes = 0.0 ;
		      double D1a = 0.0 , D1b = 0.0 , D1c = 0.0 , D1d = 0.0  ;
		      double D2a = 0.0 , D2b = 0.0 , D2c = 0.0 , D2d = 0.0  ;
		      
		      for(int q = 0 ; q < 40 ; q++){
		    	  if(testDecision[q][0].equals(datatest[q][6])) //D1:TP + TN
		    		  cor_D1++ ;
		    	  if(testDecision[q][1].equals(datatest[q][7])) //D2:TP + TN
		    		  cor_D2++ ;
		    	  if(testDecision[q][0].equals(datatest[q][6]) && testDecision[q][0].equals("yes")) //D1TP
		    		  D1a++ ;
		    	  if(testDecision[q][1].equals(datatest[q][7]) && testDecision[q][1].equals("yes")) //D2:TP
		    		  D2a++ ;
		    	  if(testDecision[q][0].equals(datatest[q][6]) && testDecision[q][0].equals("no")) //D1:TN
		    		  D1d++ ;
		    	  if(testDecision[q][1].equals(datatest[q][7]) && testDecision[q][1].equals("no")) //D2:TN
		    		  D2d++ ;
		    	  if(testDecision[q][0].equals("yes") && datatest[q][6].equals("no")) //D1:FP
		    		  D1c++ ;
		    	  if(testDecision[q][1].equals("yes") && datatest[q][7].equals("no")) //D2:FP
		    		  D2c++ ;
		    	  if(testDecision[q][0].equals("no") && datatest[q][6].equals("yes")) //D1:FN
		    		  D1b++ ;
		    	  if(testDecision[q][1].equals("no") && datatest[q][7].equals("yes")) //D1:FN
		    		  D2b++ ;
		      }
		      ans_D1 = cor_D1 / 40 ;
		      ans_D2 = cor_D2 / 40 ;

		      System.out.println("D1正確值:" + ans_D1) ;
		      System.out.println("D2正確值:" + ans_D2) ;
		      System.out.println("D1精確值:" + D1a/(D1a+D1c)) ;
		      System.out.println("D2精確值:" + D2a/(D2a+D2c)) ;
		      System.out.println("D1回傳值:" + D1a/(D1a+D1b)) ;
		      System.out.println("D2回傳值:" + D2a/(D2a+D2b)) ;
		      System.out.println("D1錯誤值:" + (1.0-ans_D1)) ;
		      System.out.println("D2錯誤值:" + (1.0-ans_D2)) ;
		      System.out.println("D1_F-measure值:" + 2*D1a/(2*D1a+D1b+D1c)) ;
		      System.out.println("D2_F-measure值:" + 2*D2a/(2*D2a+D2b+D2c)) ;
		      

		    
	}

	private static double[][] sample_mean_variance(String[][] rand) {
		double[][] a1 = new double[4][2] ; 
		double[] a1_initdub = new double[80] ;
		double meanD1yes = 0.0 , meanD1no = 0.0 , meanD2yes = 0.0 , meanD2no = 0.0 ;
		double varD1yes = 0.0 , varD2yes = 0.0 , varD1no = 0.0 , varD2no = 0.0;
		double temp = 0.0 ;
		double D1yes_a1[] = new double[80] ;
		double D1no_a1[] = new double[80] ;
		double D2yes_a1[] = new double[80] ;
		double D2no_a1[] = new double[80] ;
		int D1yes = 0 , D1no = 0 , D2yes = 0 , D2no = 0 ; //yes數量 , no數量
		
		for(int a = 0 ; a < 80 ; a++){
			temp = dubdigit(rand[a][0]) ;//把所有值轉換
			a1_initdub[a] = temp ; 
			if(rand[a][6].equals("yes")){
				D1yes_a1[D1yes] = temp ;
				D1yes++ ;
				meanD1yes = meanD1yes + temp ;
			}
			if(rand[a][6].equals("no")){
				D1no_a1[D1no] = temp ;
				D1no++ ;
				meanD1no = meanD1no + temp ;
			}
			if(rand[a][7].equals("yes")){
				D2yes_a1[D2yes] = temp ;
				D2yes++ ;
				meanD2yes = meanD2yes + temp ;
			}
			if(rand[a][7].equals("no")){
				D2no_a1[D2no] = temp ;
				D2no++ ;
				meanD2no = meanD2no + temp ;
			}
		}
		
		meanD1yes = meanD1yes/D1yes ;
		meanD1no = meanD1no/D1no ;
		meanD2yes = meanD2yes/D2yes ;
		meanD2no = meanD2no/D2no ;
		varD1yes = getvariance(meanD1yes,D1yes_a1) ;
		varD1no = getvariance(meanD1no,D1no_a1) ;
		varD2yes = getvariance(meanD2yes,D2yes_a1) ;
		varD2no = getvariance(meanD2no,D2no_a1) ;
		
		System.out.println("D1Ymean值:"+meanD1yes) ;
		System.out.println("D1Nmean值:"+meanD1no) ;
		System.out.println("D2Ymean值:"+meanD2yes) ;
		System.out.println("D2Nmean值:"+meanD2no) ;
		System.out.println("D1Yvariance值:"+varD1yes) ;
		System.out.println("D1Nvariance值:"+varD1no) ;
		System.out.println("D2Yvariance值:"+varD2yes) ;
		System.out.println("D2Nvariance值:"+varD2no) ;
		
		
		a1[0][0] = meanD1yes ; a1[0][1] = varD1yes ; //P(a1=x.x|D1=Y)
		a1[1][0] = meanD1no ;  a1[1][1] = varD1no ; //P(a1=x.x|D1=N)
		a1[2][0] = meanD2yes ; a1[2][1] = varD2yes ; //P(a1=x.x|D2=Y)
		a1[3][0] = meanD2no ;  a1[3][1] = varD2no ; //P(a1=x.x|D2=N)
		
		return a1 ;
	}

	private static double getProbabilities(double d, double mean, double var) {
		double execute = 0.0 ;
		execute = (1/(Math.sqrt(2*Math.PI*var)))*(Math.exp(-(((d-mean)*(d-mean))/(2*var)))) ;
		return execute ;
	}
	
	private static double getvariance(double mean, double[] a1) {
		double var = 0.0 ;
		int c = 0 ;
		while( a1[c] > 0 ){
			var = var + (mean-a1[c])*(mean-a1[c]) ;
			c++ ;
		}
		var = var / (c-1) ;
		return var ;
	}

	private static double dubdigit(String todub) { //StringtoDouble
		double a = 0 , digit = 0;
		String[] temprature = todub.split(",");
		
		for(String temp:temprature){
			a = Double.parseDouble(temp) ;
			if(a < 10){
				a = a/10 ;
			}
		    digit = a + digit ;
		    
		}
		return digit ;
	}

	private static double[][] event_probability(String[][] rnd) { //i表示屬性位置 
		double[][] p = new double[8][6] ;//a2~5的機率值
		int D1Ynum = 0 ;
		int D1Nnum = 0 ;
		int D2Ynum = 0 ;
		int D2Nnum = 0 ;
		int m = 0 ;
		double mp = 1/80 ;
		double [][]np = new double[2][6] ;
		
		for(int a = 0 ; a < 80 ; a++){
			if(rnd[a][6].equals("yes")){//D1=Y
				D1Ynum++ ;
			}
			if(rnd[a][6].equals("no")){//D1=n
				D1Nnum++ ;
			}
			if(rnd[a][7].equals("yes")){//D2=Y
				D2Ynum++ ;
			}
			if(rnd[a][7].equals("no")){//D2=n
				D2Nnum++ ;
			}
			
			if(rnd[a][1].equals("yes")){
				np[0][1]++ ;
			}
			if(rnd[a][2].equals("yes")){
				np[0][2]++ ;
			}
			if(rnd[a][3].equals("yes")){
				np[0][3]++ ;
			}
			if(rnd[a][4].equals("yes")){
				np[0][4]++ ;
			}
			if(rnd[a][5].equals("yes")){
				np[0][5]++ ;
			}

			if(rnd[a][1].equals("no")){
				np[1][1]++ ;
			}
			if(rnd[a][2].equals("no")){
				np[1][2]++ ;
			}
			if(rnd[a][3].equals("no")){
				np[1][3]++ ;
			}
			if(rnd[a][4].equals("no")){
				np[1][4]++ ;
			}
			if(rnd[a][5].equals("no")){
				np[1][5]++ ;
			}
		}
		
		for(int l = 1 ; l < 6 ; l++){ //a2~a5
			for(int c = 0 ; c < 80 ; c++){
				if(rnd[c][6].equals("yes")&&rnd[c][l].equals("yes")){ //D1=yes^ax=yes
					p[0][l]++ ;
				}
				if(rnd[c][6].equals("yes")&&rnd[c][l].equals("no")){ //D1=yes^ax=no
					p[1][l]++ ;
				}
				if(rnd[c][6].equals("no")&&rnd[c][l].equals("yes")){ //D1=no^ax=yes
					p[2][l]++ ;
				}
				if(rnd[c][6].equals("no")&&rnd[c][l].equals("no")){ //D1=no^ax=no
					p[3][l]++ ;
				}
				if(rnd[c][7].equals("yes")&&rnd[c][l].equals("yes")){ //D2=yes^ax=yes
					p[4][l]++ ;
				}
				if(rnd[c][7].equals("yes")&&rnd[c][l].equals("no")){ //D2=yes^ax=no
					p[5][l]++ ;
				}
				if(rnd[c][7].equals("no")&&rnd[c][l].equals("yes")){ //D2=no^ax=yes
					p[6][l]++ ;
				}
				if(rnd[c][7].equals("no")&&rnd[c][l].equals("no")){ //D2=no^ax=no
					p[7][l]++ ;
				}
				
			}
		}
		for(int l = 0 ; l < 4 ; l++){ //D1
			for(int c = 1 ; c < 6 ; c++){
				if(l<2){
					if(l==0)
						p[l][c] = (p[l][c]+m*np[0][c]*mp)/(D1Ynum+m) ;
					else if(l==1)
						p[l][c] = (p[l][c]+m*np[1][c]*mp)/(D1Ynum+m) ;
						
				}
				else if(l>=2){
					if(l==2)
						p[l][c] = (p[l][c]+m*np[0][c]*mp)/(D1Nnum+m) ;
					else if(l==3)
						p[l][c] = (p[l][c]+m*np[1][c]*mp)/(D1Nnum+m) ;
				}	
			}
		}
		for(int l = 4 ; l < 8 ; l++){ //D2
			for(int c = 1 ; c < 6 ; c++){
				if(l<6){
					if(l==4)
						p[l][c] = (p[l][c]+m)/(D2Ynum+m) ;
					else if(l==5)
						p[l][c] = (p[l][c]+m)/(D2Ynum+m) ;
				}
				else if(l>=6){
					if(l==6)
						p[l][c] = (p[l][c]+m*np[0][c]*mp)/(D2Nnum+m) ;
					else if(l==7)
						p[l][c] = (p[l][c]+m*np[1][c]*mp)/(D2Nnum+m) ;
				}
					
			}
		}
		return p ;
		
	}

	private static String[][] random_80sample(String[][] data) {
		String[][] temp = new String[80][8] ;
		int rnd ;
		HashSet rndSet = new HashSet<Integer>(80) ;
		for(int i = 0 ; i < 80 ; i++){
			rnd = (int)(120*Math.random());
			while(!rndSet.add(rnd))
				rnd = (int)(120*Math.random());
			temp[i] = data[rnd] ;
			
		}
		return temp ;
		
	}
	
	private static String[][] random_40sample(String[][] data) {
		String[][] temp = new String[40][8] ;
		int rnd ;
		HashSet rndSet = new HashSet<Integer>(40) ;
		for(int i = 0 ; i < 40 ; i++){
			rnd = (int)(120*Math.random());
			while(!rndSet.add(rnd))
				rnd = (int)(120*Math.random());
			temp[i] = data[rnd] ;
			
		}
		return temp ;
		
	}

	private static String D1_Test_model(double a1, String a2, String a3, String a4, String a5, String a6, double[][] p, double[][] a1_init) {
		double Pn,Py ;
		double Pa2 = 0.0 , Pa3 = 0.0 , Pa4 = 0.0 , Pa5 = 0.0 , Pa6 = 0.0 ;
		//P(Rule|Y)
		if(a2.equals("yes")){
			Pa2 = p[0][1];
		}
		else if(a2.equals("no")){
			Pa2 = p[1][1];
		}
		
		if(a3.equals("yes")){
			Pa3 = p[0][2];
		}
		else if(a3.equals("no")){
			Pa3 = p[1][2];
		}
		
		if(a4.equals("yes")){
			Pa4 = p[0][3];
		}
		else if(a4.equals("no")){
			Pa4 = p[1][3];
		}
		
		if(a5.equals("yes")){
			Pa5 = p[0][4];
		}
		else if(a5.equals("no")){
			Pa5 = p[1][4];
		}
		
		if(a6.equals("yes")){
			Pa6 = p[0][5];
		}
		else if(a6.equals("no")){
			Pa6 = p[1][5];
		}
		
		Py = getProbabilities(a1,a1_init[0][0],a1_init[0][1])*Pa2*Pa3*Pa4*Pa5*Pa6 ;
		
		//P(Rule|N)
		if(a2.equals("yes")){
			Pa2 = p[2][1];
		}
		else if(a2.equals("no")){
			Pa2 = p[3][1];
		}
		
		if(a3.equals("yes")){
			Pa3 = p[2][2];
		}
		else if(a3.equals("no")){
			Pa3 = p[3][2];
		}
		
		if(a4.equals("yes")){
			Pa4 = p[2][3];
		}
		else if(a4.equals("no")){
			Pa4 = p[3][3];
		}
		
		if(a5.equals("yes")){
			Pa5 = p[2][4];
		}
		else if(a5.equals("no")){
			Pa5 = p[3][4];
		}
		
		if(a6.equals("yes")){
			Pa6 = p[2][5];
		}
		else if(a6.equals("no")){
			Pa6 = p[3][5];
		}
		
		Pn = getProbabilities(a1,a1_init[1][0],a1_init[1][1])*Pa2*Pa3*Pa4*Pa5*Pa6 ;
		
		if(Py>Pn){
			return "yes" ;
		}
		else
			return "no" ;
		
	}
	private static String D2_Test_model(double a1, String a2, String a3, String a4, String a5, String a6, double[][] p, double[][] a1_init) {
		double Pn,Py ;
		double Pa2 = 0.0 , Pa3 = 0.0 , Pa4 = 0.0 , Pa5 = 0.0 , Pa6 = 0.0 ;
		//P(Rule|Y)
		if(a2.equals("yes")){
			Pa2 = p[4][1];
		}
		else if(a2.equals("no")){
			Pa2 = p[5][1];
		}
		
		if(a3.equals("yes")){
			Pa3 = p[4][2];
		}
		else if(a3.equals("no")){
			Pa3 = p[5][2];
		}
		
		if(a4.equals("yes")){
			Pa4 = p[4][3];
		}
		else if(a4.equals("no")){
			Pa4 = p[5][3];
		}
		
		if(a5.equals("yes")){
			Pa5 = p[4][4];
		}
		else if(a5.equals("no")){
			Pa5 = p[5][4];
		}
		
		if(a6.equals("yes")){
			Pa6 = p[4][5];
		}
		else if(a6.equals("no")){
			Pa6 = p[5][5];
		}
		
		Py = getProbabilities(a1,a1_init[2][0],a1_init[2][1])*Pa2*Pa3*Pa4*Pa5*Pa6 ;
		
		//P(Rule|N)
		if(a2.equals("yes")){
			Pa2 = p[6][1];
		}
		else if(a2.equals("no")){
			Pa2 = p[7][1];
		}
		
		if(a3.equals("yes")){
			Pa3 = p[6][2];
		}
		else if(a3.equals("no")){
			Pa3 = p[7][2];
		}
		
		if(a4.equals("yes")){
			Pa4 = p[6][3];
		}
		else if(a4.equals("no")){
			Pa4 = p[7][3];
		}
		
		if(a5.equals("yes")){
			Pa5 = p[6][4];
		}
		else if(a5.equals("no")){
			Pa5 = p[7][4];
		}
		
		if(a6.equals("yes")){
			Pa6 = p[6][5];
		}
		else if(a6.equals("no")){
			Pa6 = p[7][5];
		}
		
		Pn = getProbabilities(a1,a1_init[3][0],a1_init[3][1])*Pa2*Pa3*Pa4*Pa5*Pa6 ;
		
		if(Py>Pn){
			return "yes" ;
		}
		else
			return "no" ;
	}


}
