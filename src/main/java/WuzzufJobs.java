import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.*;
import org.knowm.xchart.*;

import org.knowm.xchart.style.Styler;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;


public class WuzzufJobs {

    public static void main(String[] args) throws IOException {

        // Static stuff:
        Logger.getLogger ("org").setLevel (Level.ERROR);
        // Create Spark Session to create connection to Spark
        final SparkSession sparkSession = SparkSession.builder().appName("Wuzzuf Jobs Data")
                .master("local[4]").getOrCreate();

        // Get DataFrameReader using SparkSession and set header option to true
        // to specify that first row in file contains name of columns
        final DataFrameReader dataFrameReader = sparkSession.read().option("header", true);
        final Dataset<Row> jobsData = dataFrameReader.csv("src/main/resources/Wuzzuf_Jobs.csv");

        // Methods:

        // 1) Schema and sample
        // ============================================================================================================
        // Print Schema to see column names, types and other metadata
        jobsData.printSchema();

        // Show top 20 rows in dataframe
        jobsData.show(20);

        // 2) Statistics
        // ============================================================================================================
        // Show Summary Statistics
        jobsData.describe().show();

        // 3) Clean data
        // ============================================================================================================
        // Drop rows with null values
        Dataset<Row> cleanedJobsData = jobsData.na().drop();
        cleanedJobsData.describe().show();

        // Drop duplicate rows
        cleanedJobsData = jobsData.dropDuplicates();
        cleanedJobsData.describe().show();

        // 4) Most companies with job posts
        // ============================================================================================================
        // Count of Jobs per company
        cleanedJobsData.createOrReplaceTempView ("JOBS_DATA");
        Dataset<Row> jobCountPerCompany = sparkSession.sql("SELECT Company, count(Title) as freq "+
                                                                  "FROM JOBS_DATA GROUP BY Company "+
                                                                  "ORDER BY freq DESC ");
        jobCountPerCompany.describe().show();
        jobCountPerCompany.show();

        // 5)
        // ============================================================================================================
        // Prepare data for pie chart
        List<String> companyNames = jobCountPerCompany.select("Company").limit(10).as(Encoders.STRING()).collectAsList();
        List<Long> jobCount = jobCountPerCompany.select("freq").limit(10).as(Encoders.LONG()).collectAsList();

        // Make pie chart of above data
        PieChart pieChart1 = new PieChartBuilder().width(1280).height(800).title("Jobs Per Company").build();
      
        // Customize Chart
        pieChart1.getStyler ().setLegendPosition (Styler.LegendPosition.InsideNW);
        
        for (int i=0; i<companyNames.size(); i++){
            pieChart1.addSeries(companyNames.get(i), jobCount.get(i));
        }
        BitmapEncoder.saveBitmap(pieChart1, "src/main/resources/jobCountPerCompany", BitmapEncoder.BitmapFormat.PNG);
        // new SwingWrapper(pieChart1).displayChart ();

        // 6)
        // ============================================================================================================
        // Most popular title
        Dataset<Row> popularTitles = sparkSession.sql("SELECT Title, count(Title) as freq "+
                                                             "FROM JOBS_DATA GROUP BY Title "+
                                                             "ORDER BY freq DESC ");
        popularTitles.show();

        // ============================================================================================================

        // 7)
        // Prepare data for bar chart
        List<String> jobTitles = popularTitles.select("Title").limit(10).as(Encoders.STRING()).collectAsList();
        List<Long> jobCount2 = popularTitles.select("freq").limit(10).as(Encoders.LONG()).collectAsList();

        // Make bar chart of above data
        CategoryChart barChart = new CategoryChartBuilder().width(1280).height(800).title("Popular Job Titles").build();
      
        // Customize Chart
        barChart.getStyler ().setLegendPosition (Styler.LegendPosition.InsideNW);
        barChart.getStyler ().setHasAnnotations (true);
        barChart.getStyler ().setStacked (true);
      
        barChart.addSeries("Popular Job Titles", jobTitles, jobCount2);
        
        BitmapEncoder.saveBitmap(barChart, "src/main/resources/popularTitles", BitmapEncoder.BitmapFormat.PNG);

        // 8)
        // ============================================================================================================
        // Most popular areas
        Dataset<Row> popularLocations = sparkSession.sql("SELECT Location, count(Location) as freq "+
                                                                "FROM JOBS_DATA GROUP BY Location "+
                                                                "ORDER BY freq DESC ");
        popularLocations.show();

        // 9)
        // Prepare data for bar chart
        List<String> locations = popularLocations.select("Location").limit(10).as(Encoders.STRING()).collectAsList();
        List<Long> locCount = popularTitles.select("freq").limit(10).as(Encoders.LONG()).collectAsList();

        // Make bar chart of above data
        CategoryChart barChart2 = new CategoryChartBuilder().width(1280).height(800).title("Popular Job Locations").build();
   
        // Customize Chart
        barChart2.getStyler ().setLegendPosition (Styler.LegendPosition.InsideNW);
        barChart2.getStyler ().setHasAnnotations (true);
        barChart2.getStyler ().setStacked (true);
     
        barChart2.addSeries("Popular Job Locations", locations, locCount);
        
        BitmapEncoder.saveBitmap(barChart2, "src/main/resources/popularLocations", BitmapEncoder.BitmapFormat.PNG);

        // 10)
        // ============================================================================================================
        // Most required skills
        List<String> skills = cleanedJobsData.select("Skills").map(row -> row.getString(0), Encoders.STRING()).collectAsList();
        Dataset<Row> popularSkills = cleanedJobsData.select("Skills")
                .flatMap(row -> Arrays.asList(row.getString(0).split(",")).iterator(), Encoders.STRING())
                .filter(s -> !s.isEmpty())
                .map(word -> new Tuple2<>(word.toLowerCase(), 1L), Encoders.tuple(Encoders.STRING(), Encoders.LONG()))
                .toDF("word", "count")
                .groupBy("word")
                .sum("count").orderBy(new Column("sum(count)").desc()).withColumnRenamed("sum(count)", "cnt");
        popularSkills.show();
        // ============================================================================================================
    
        
        // 11) Factorize the YearsExp feature
       
        // 12) Apply K-means for job title and companies
        // ============================================================================================================
        // Cluster the data into two classes using KMeans
        int numOfClusters = 2;
        int numOfIterations = 20;
        Vector companyFeature = (Vector)cleanedJobsData.select("Company");
        
        // Trains a k-means model
        KMeansModel clusters = KMeans.train((RDD<Vector>) companyFeature, numOfClusters, numOfIterations);
        // Shows the result
        Vector[] centers = clusters.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }
        
        // ============================================================================================================
       
    }

}

