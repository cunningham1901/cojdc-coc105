import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.InputSampler;
import org.apache.hadoop.mapreduce.lib.partition.TotalOrderPartitioner;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class NGrams extends Configured implements Tool {

    public static class NGMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text ngram = new Text();

        //Compile regex now, to save time
        private static final Pattern punctuation = Pattern.compile("\\p{Punct}");
        private static final Pattern spaces = Pattern.compile("\\s+");

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            //Split string into array, removing punctuation & whitespace
            String[] words = spaces.split(punctuation.matcher(value.toString()).replaceAll(" ").trim().toLowerCase());

            //Get n from cmd line argument e.g.: -D ngram.n=2
            Configuration conf = context.getConfiguration();
            int n = conf.getInt("ngram.n", 2);

            // Loop through line, if not possible to create any n-grams (i.e. line.length < n) then skip line
            StringBuilder ngram_string = new StringBuilder();  // Reusing string builder is faster
            for (int i=0; i<(words.length-(n-1)); i++) {
                ngram_string.setLength(0);  // Clear old string
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

    public static class NGFileOffset implements WritableComparable {
        private long offset;
        private String file;

        @Override
        public int compareTo(Object o) {
            NGFileOffset that = (NGFileOffset)o;
            int fileComparison = this.file.compareTo(that.file);
            if (fileComparison == 0) {
                return (int)Math.signum((double)(this.offset - that.offset));
            }
            return fileComparison;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeLong(offset);
            Text.writeString(dataOutput, file);

        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            this.offset = dataInput.readLong();
            this.file = Text.readString(dataInput);
        }
    }

    public static class NGCombineFileInputFormat extends CombineFileInputFormat<NGFileOffset, Text> {
        public RecordReader<NGFileOffset, Text> createRecordReader(InputSplit split, TaskAttemptContext context) throws IOException {
            return new CombineFileRecordReader<NGFileOffset, Text>(
                    (CombineFileSplit)split, context, NGCombineRecordReader.class);
        }
    }

    public static class NGCombineRecordReader extends RecordReader<NGFileOffset, Text> {
        private long startOffset; //offset of the chunk;
        private long end; //end of the chunk;
        private long pos; // current pos
        private FileSystem fs;
        private Path path;
        private NGFileOffset key;
        private Text value;

        private FSDataInputStream fileIn;
        private LineReader reader;

        public NGCombineRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer index) throws IOException {
            this.path = split.getPath(index);
            fs = this.path.getFileSystem(context.getConfiguration());
            this.startOffset = split.getOffset(index);
            this.end = startOffset + split.getLength(index);
            boolean skipFirstLine = false;

            //open the file
            fileIn = fs.open(path);
            if (startOffset != 0) {
                skipFirstLine = true;
                --startOffset;
                fileIn.seek(startOffset);
            }
            reader = new LineReader(fileIn);
            if (skipFirstLine) {  // skip first line and re-establish "startOffset".
                startOffset += reader.readLine(new Text(), 0,
                        (int)Math.min((long)Integer.MAX_VALUE, end - startOffset));
            }
            this.pos = startOffset;
        }

        public void initialize(InputSplit split, TaskAttemptContext context)
                throws IOException, InterruptedException {
            //Not called, uses custom constructor
        }

        public void close() throws IOException { }

        public float getProgress() throws IOException {
            if (startOffset == end) {
                return 0.0f;
            } else {
                return Math.min(1.0f, (pos - startOffset) / (float)(end - startOffset));
            }
        }

        public boolean nextKeyValue() throws IOException {
            if (key == null) {
                key = new NGFileOffset();
                key.file = path.getName();
            }
            key.offset = pos;
            if (value == null) {
                value = new Text();
            }
            int newSize = 0;
            if (pos < end) {
                newSize = reader.readLine(value);
                pos += newSize;
            }
            if (newSize == 0) {
                key = null;
                value = null;
                return false;
            } else {
                return true;
            }
        }

        public NGFileOffset getCurrentKey() throws IOException, InterruptedException {
            return key;
        }

        public Text getCurrentValue() throws IOException, InterruptedException {
            return value;
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


        //Input format is set to combine files for speed, if desired
        if (conf.getBoolean("ngrams.combineInputFiles", false)) {
            job.setInputFormatClass(NGCombineFileInputFormat.class);
        }
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
                System.out.println("Output will be sorted globally by key");
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