import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.InputSampler;
import org.apache.hadoop.mapreduce.lib.partition.TotalOrderPartitioner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class NGrams extends Configured implements Tool {

    public static class NGMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text ngram = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            //Split string into array, removing punctuation & whitespace
            String[] words = value.toString().replaceAll("\\p{Punct}", " ").trim().split("\\s+");

            //Get n from cmd line argument e.g.: -D ngram.n=2
            Configuration conf = context.getConfiguration();
            int n = conf.getInt("ngram.n", 2);

            // Loop through line, if not possible to create any n-grams (i.e. line.length < n) then skip line
            for (int i=0; i<(words.length-(n-1)); i++) {
                StringBuilder ngram_string = new StringBuilder();
                //Build the n-gram
                for (int j=0; j<n; j++) ngram_string.append(words[i + j]).append(" ");
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

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = this.getConf();
        System.out.println("The NGram N parameter is:" + conf.get("ngram.n", "2"));
        Job job = new Job(conf, "NGrams");
        job.setJarByClass(NGrams.class);
        job.setMapperClass(NGMapper.class);
        job.setReducerClass(NGReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        if (conf.getBoolean("ngram.sort.global", false)) {
            System.out.println("Output will be sorted globally across reducers");
            job.setPartitionerClass(TotalOrderPartitioner.class);
            InputSampler.Sampler<Object, Text> sampler = new InputSampler.RandomSampler<>(0.1, 2000);
            InputSampler.writePartitionFile(job, sampler);
        }

        // Execute job and return status
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new NGrams(), args);
        System.exit(res);
    }
}