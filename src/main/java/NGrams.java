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
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
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
        Path inputPath = new Path(args[0]);
        Path unsortedOutputPath = new Path(args[1]);
        Path partitionPath = new Path(args[2]);
        Path sortedOutputPath = new Path(args[3]);

        Configuration conf = this.getConf();
        System.out.println("The NGram N parameter is:" + conf.get("ngram.n", "2"));
        Job job = new Job(conf, "NGrams");
        job.setJarByClass(NGrams.class);
        job.setMapperClass(NGMapper.class);
        job.setReducerClass(NGReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, unsortedOutputPath);

        //output format set here as seqfile for input to next job
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        int exit_code = job.waitForCompletion(true) ? 0 : 1;

        if (exit_code == 0) {
            //Create a job for sorting
            Job sortJob = new Job(conf, "Ngrams - Sorting");
            sortJob.setJarByClass(NGrams.class);
            sortJob.setMapperClass(Mapper.class);
            sortJob.setReducerClass(Reducer.class);
            sortJob.setMapOutputKeyClass(Text.class);
            sortJob.setMapOutputValueClass(IntWritable.class);



            sortJob.setInputFormatClass(SequenceFileInputFormat.class);
            SequenceFileInputFormat.setInputPaths(sortJob, unsortedOutputPath);
            sortJob.setOutputFormatClass(TextOutputFormat.class);
            TextOutputFormat.setOutputPath(sortJob, sortedOutputPath);

            if (conf.get("ngram.sort", "none").equals("globalKey")) {
                TotalOrderPartitioner.setPartitionFile(sortJob.getConfiguration(), partitionPath);
                InputSampler.Sampler<Text, IntWritable> sampler = new InputSampler.RandomSampler<>(0.01, 1000);
                InputSampler.writePartitionFile(sortJob, sampler);
                sortJob.setPartitionerClass(TotalOrderPartitioner.class);
            }

            // Execute job and return status
            exit_code = sortJob.waitForCompletion(true) ? 0:1;
        }

        return exit_code;

    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new NGrams(), args);
        System.exit(res);
    }
}