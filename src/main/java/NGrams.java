import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class NGrams {

    public static class NGMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text ngram = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            //Split string into array, removing punctuation & whitespace
            String[] words = value.toString().replaceAll("\\p{Punct}", " ").trim().split("\\s+");

            //Get n from cmd line argument e.g.: -D ngram_n=2
            Configuration conf = context.getConfiguration();
            int n = conf.getInt("ngram_n", 2);

            // Loop through line, if not possible to create any n-grams (i.e. line.length < n) then skip line
            for (int i=0; i<(words.length-(n-1)); i++) {
                StringBuilder ngram_string = new StringBuilder();
                //Build the n-gram
                for (int j=0; j<n; j++) {
                    ngram_string.append(words[i + j]).append(" ");
                }
                ngram_string.deleteCharAt(ngram_string.length()-1);
                ngram.set(ngram_string.toString());
                context.write(ngram, one);
            }
        }
    }

    public static class NGReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Ngrams");
        job.setJarByClass(NGrams.class);
        job.setMapperClass(NGMapper.class);
        job.setReducerClass(NGReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}