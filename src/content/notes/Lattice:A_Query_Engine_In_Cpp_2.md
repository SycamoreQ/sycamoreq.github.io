---
title: "Lattice: A Query Engine in C++ - Part 2"
date: 2026-11-18
description: "This blog is the second part of a series where I implement a query engine from scratch."
tags: [QueryEngine, C++]
draft: false
---

# Lattice: A Query Engine in C++ - Part 2

* In the last blog, we looked into the very basic data representation that will be used in Lattice. Now that we have our data, we need to move to the next step of any Query Engine and that is to formalize various data sources as well as look into building the first iteration of a query plan: a `logical plan`.

* Even in the last blog, I attributed the process of creating a Query Engine to building a compiler. For all you compiler folks, this is probably where the Query Engine decides to convert the given data/query into the data representation we created as well as convert it into a tree structure that represents the computation without specifying exactly how to execute it. That is called a `logical-plan`. This is then passed down to other modules that will actually handle where to execute and so on.

* As always, the repository and the code is [here](https://github.com/SycamoreQ/lattice)

## Data Sources

* We construct a `data-source` interface for the sole reason that is to also specify *where* data is present i.e what type of file format. In Lattice (and KQuery consequentially) I specify 3 sources - CSV, Parquet and In-Memory. I think the first two is pretty well known data formats, In-Memory is a sort of generalized data format which represent some data that is intermediary between query execution, essentially some scratchpad data formats.

* I will not be going through all 3 interfaces, but just one which is the `CSV`. I think it was a pretty good ride coding this because there is no C++ STL code for parsing `CSV` directly. So I had to make one from scratch.

* We actually start of by defining a general class which will be overridden by the 3 files, a pattern similar to what we had followed in the last blog.

```c++
class RecordBatchIterator {
public:
    virtual ~RecordBatchIterator() = default;

    // Returns true if there is another batch available
    virtual bool hasNext() = 0;

    // Returns the next batch
    virtual RecordBatch next() = 0;
};

class DataSource {
public:
    virtual ~DataSource() = default;
    virtual Schema schema() const = 0;

    // Now returns a lazy iterator instead of a giant vector
    virtual std::shared_ptr<RecordBatchIterator> scan(const std::vector<std::string>& projection) = 0;
};
```

* We have two classes, one is the `RecordBatchIterator` which is pretty important. We use a `RecordBatchIterator` to enable lazy, streaming execution, allowing the query engine to process datasets much larger than your available RAM. If `scan()` returned a `std::vector<RecordBatch>`, the engine would have to read the entire file into memory before doing any work. The iterator solves three critical database problems:

  * **Out-of-Memory (OOM) Prevention**: A 50GB CSV file would crash our program if loaded at
    once into a vector/custom data representation. The iterator pulls data from disk one chunk
    at a time (e.g., the 1024 row `batchSize` we configured in the last blog). The memory
    footprint remains stable regardless of how large the underlying file is.

  * **Pipelined Execution**: Downstream operators don't have to wait for the disk scan to
    finish. As soon as the DataSource yields the first batch via `next()`, the engine can
    immediately start applying Filter or Projection logic to it while subsequent batches are
    still waiting to be read from disk.

  * **Early Termination (Short-circuiting)**: If a user runs a query with a `LIMIT 10` clause,
    the execution plan only calls `next()` once. It processes the first batch, fulfills the limit,
    and destroys the iterator. If scan() returned a vector, it would read millions of rows from
    disk just to throw 99.9% of them away.

* Coming to the `DataSource` class, we have a schema defined through `schema()` as well as a `scan()` function as illustrated above on its importance. The combination of these classes ensure that data is streamed and not loaded at once in order to have things overlap (pipelining) as well as rollback options.

### The CSV Source

* Our first task is to override the `RecordBatchIterator`. We will call it `CSVReaderIterator`, and this was probably the harder part of doing this. We define two helper functions `nextBatch()` and `createBatch()` which we will use later when we define `hasNext()` and `next()`.

* `nextBatch()` is a pretty simple function, it just iterates every `row` of CSV data and parses it. Just the very internals of an iterator. If there is no data to be parsed, it will just return a `nullopt`.

* `createBatch()` is where the crucial "pivot" operation happens. It takes the row-oriented, plain-text data from the CSV file and converts it into strongly-typed, column-oriented Apache Arrow arrays.

* Because Apache Arrow arrays are immutable (read-only) once created, you cannot just create an empty array and add items to it. This function uses "Builders" to accumulate the data first, breaking the process down into three main steps:

  1. **Creating the Builders**: For every field in the schema, it creates a corresponding
     Apache Arrow `ArrayBuilder`. These builders pre-allocate memory and wait to receive
     data top-to-bottom.

  2. **Parsing and Appending**: It iterates through the cells, trims whitespace, and handles
     empty cells by appending nulls. It then uses a `switch` statement to safely downcast the
     generic builder (e.g., to `arrow::Int32Builder*`) and parses the string to the correct
     C++ type (like `std::stoi`). It includes a `try/catch` block as a safety net—if a dirty
     row exists, it gracefully inserts a null instead of crashing the query engine.

  3. **Finalizing the Arrays**: Once all rows are processed and the builders are full, it
     calls `Finish()` on each one. This seals the memory layout into highly optimized,
     immutable `arrow::Array` objects, which are then wrapped into our custom `RecordBatch`
     to be passed up the execution plan.

* With the helper methods out of the way, `hasNext()` and `next()` just manage a simple state machine. `hasNext()` checks if we've started; if not, it triggers the first read. `next()` consumes the current batch and immediately queues up the next one by calling `nextBatch()`.

### The CsvDataSource Class

* Now we need the actual factory that configures the parser and gives us this iterator: the `CsvDataSource`.

```c++
class CsvDataSource : public DataSource {
// ... member variables omitted for brevity ...
    Schema inferSchema() const {
        std::ifstream file(filename);
        if (!file.good()) throw std::runtime_error("File not found: " + filename);

        csv::CSVFormat format = defaultSettings();
        format.header_row(hasHeaders ? 0 : -1);

        csv::CSVReader parser(filename, format);
        std::vector<std::string> headers = parser.get_col_names();

        std::vector<Field> fields;
        for (size_t i = 0; i < headers.size(); ++i) {
            std::string colName = hasHeaders ? headers[i] : "field_" + std::to_string(i + 1);
            fields.emplace_back(colName, ArrowTypes::StringType);
        }
        return Schema(fields);
    }
// ...
```

* **Schema Inference**: In strongly-typed query engines, we need to know the schema before we execute the query. If the user doesn't provide one, `inferSchema()` peeks at the CSV file. It reads the first row to extract column names (or creates dummy names like `field_1` if there are no headers) and defaults all data types to `StringType`. Downstream SQL `CAST` operations can convert these strings into integers or doubles if needed.

* **Initialization**: In Kotlin, we could use `by lazy { inferSchema() }` to defer this file read until it's actually needed. C++ lacks a direct `lazy` language feature, so we handle this cleanly in the constructor's initializer list: `finalSchema(provided_schema.has_value() ? provided_schema.value() : inferSchema())`.

* Finally, the entry point for execution: `scan()`.

```c++
    std::shared_ptr<RecordBatchIterator> scan(const std::vector<std::string>& projection) override {
        // ... file checks omitted ...
        Schema readSchema = finalSchema;
        if (!projection.empty()) {
            readSchema = finalSchema.select(projection);
        }

        csv::CSVFormat format = defaultSettings();
        format.header_row(hasHeaders ? 0 : -1);
        
        // ... handling headerless columns ...

        auto parser = std::make_unique<csv::CSVReader>(filename, format);
        return std::make_shared<CsvReaderIterator>(readSchema, std::move(parser), batchSize);
    }
```

* **Projection Pushdown**: Notice the `projection` argument. If our query is `SELECT name FROM users`, we don't want to waste CPU cycles parsing the `age`, `address`, and `phone` columns. `scan()` shrinks the `finalSchema` down to just the requested columns. We can pass this narrowed schema down to the `CsvReaderIterator`, which is a massive I/O and CPU optimization known as *projection pushdown*.

* It then initializes the third-party `csv::CSVReader` and hands it off to our `CsvReaderIterator` to begin the lazy, streaming execution.

### Parquet and In-Memory Sources

* While CSV is a great starting point for its simplicity and ubiquity, a modern query engine needs to support formats that are actually optimized for performance. This brings us to the other two data sources in Lattice: **Parquet** and **In-Memory**.

#### The Parquet Source

* If CSV is the slow, human-readable workhorse of data engineering, Parquet is the highly tuned, binary sports car. Parquet is a columnar storage file format. Because Lattice (via Apache Arrow) is built entirely around columnar memory, reading Parquet files is an incredibly natural fit.

* Unlike CSV, which requires heavy string-parsing and type-casting (as we saw in `createBatch()`), Parquet already stores data in strongly-typed columns. A Parquet `DataSource` is significantly faster because it can often map the disk's columnar layout directly into Arrow's in-memory columnar layout with minimal transformation, allowing our query engine to skip reading columns that aren't included in the query projection.

#### The In-Memory Source

* The In-Memory source is slightly different—it doesn't represent a file on your hard drive at all. Instead, it acts as a generalized logical data source for data that already resides in RAM.

* When building a query engine, you frequently generate intermediary results. If you perform a massive `JOIN` or a complex `GROUP BY`, the engine needs a place to hold that temporary data before passing it to the next stage of the execution plan (like a `LIMIT` or `ORDER BY`). The In-Memory data source acts as this "scratchpad," allowing the engine to treat these intermediate, on-the-fly datasets with the exact same `DataSource` interface and `RecordBatchIterator` rules that it uses for actual files.

## Logical Plans

* A logical plan is exactly what it is. Creating a plan for a specific query in an efficient manner so that anything that will execute the query, knows what to execute, validate, optimize and scale. Logical plans are pretty much one of the most important parts of a Query Engine and since it does so much, we would have to provide its inputs and expect a good output.

* To sort of give an example, I take it from the KQuery docs themselves. Suppose if we have a query:

```sql
SELECT name, salary * 1.1 AS new_salary
FROM employees
WHERE department = 'Engineering'
```

* Before we can execute this, we need a structured representation. The logical plan captures:

  * Read from the employees table

  * Keep only rows where department = 'Engineering'

  * Compute name and salary \* 1.1 for each remaining row

* As you can see, it breaks down the entire query, knows what to do each line of code and what each line of code would return. A logical plan builds one after the other the order of what is to be done using operators that it will define under its belt. We will see those operators now.

* As always we start out with the interface.

```c++
class LogicalPlan {
    public:

        virtual ~LogicalPlan() = default;

        /**
         * Returns the schema of the data that will be produced by this logical plan.
         */
        virtual Schema schema() const = 0;

        /**
         * Returns the children (inputs) of this logical plan. This method is used to
         * enable use of the visitor pattern to walk a query tree.
         */
        virtual std::vector<std::shared_ptr<LogicalPlan>> children() const = 0;

        virtual std::string toString() const = 0;

        virtual std::string pretty() const {
            return format(*this);
        }
};


std::string format(const LogicalPlan& plan, int indent = 0) {
    std::stringstream b;
    b << std::string(indent, '\t');
    b << plan.toString() << "\n";
    for (const auto& child : plan.children()) {
        if (child) {
            b << format(*child, indent + 1);
        }
    }

    return b.str();
}
```

* We have two core pure virtual functions: `schema()` which, as always, returns the Schema of the inherent data, and `children()`. This is interesting because it actually returns the input relations (datasets) that will be used in a logical plan.

* For example, if we want to plan a Join operation (something that combines two tables together), it will take two input arguments representing the left and right datasets, and they themselves have to be logical plans. This is very analogous to how a compiler would parse a binary operation, wherein it would classify the left and right operands as variables and walk the AST.

* I also equip the interface with a pretty printer function `pretty()` which actually prints the logical plan for us. Using the same example as before, our printed output would be:

```text
Projection: #name, #salary * 1.1 AS new_salary
  Selection: #department = 'Engineering'
    Scan: employees; projection=None
```

* Moving forward, we have actually defined the plan interface which will dictate how data moves through the Logical Plan itself. But nothing about the actual *computation* exists in the interface. In our query, we still have to define the `salary * 1.1` with some sort of interface that represents an *expression* so that the Logical Plan can use the expression to build anything. So naturally we build a `Logical Expr` interface:

```c++
/**
 * Logical Expression for use in logical query plans. The logical expression
 * provides information needed during the planning phase such as the name and
 * data type of the expression.
 */
class LogicalExpr {
public:
    virtual ~LogicalExpr() = default;

    /**
     * Return meta-data about the value that will be produced by this expression
     * when evaluated against a particular input.
     *
     * Marked 'const' because extracting metadata should not modify the expression.
     */
    virtual Field toField(const LogicalPlan& input) const = 0;

    virtual std::string toString() const = 0;
};
```

* I have this table that summarizes the expressions we will be having in Lattice as a whole:

Expressions can be simple (a column reference, a literal value) or complex (nested arithmetic, function calls). Here are common expression types:

| Expression Type | Examples | 
| ----- | ----- | 
| Literal Value | "hello", 12.34, true | 
| Column Reference | user_id, first_name, salary | 
| Math Expression | salary \* 0.1, price + tax | 
| Comparison Expression | age >= 21, status != 'inactive' | 
| Boolean Expression | age >= 21 AND country = 'US' | 
| Aggregate Expression | MIN(salary), MAX(salary), SUM(amount), COUNT(\*) | 
| Scalar Function | UPPER(name), CONCAT(first_name, ' ', last_name) | 
| Aliased Expression | salary \* 1.1 AS new_salary | 

* Expressions then form trees, just like the AST for a compiler. Given an expression like `(a + b) * c`, it has a multiply at the root with two children: an add expression (with children `a` and `b`) and a column reference `c`.

* The interface has one function `toField()` which returns the name and data type of the expression's output. Nothing serious.

* We now go into defining each type of expression as mentioned in the table. They all will of course inherit the `LogicalExpr` class.

### Column Expression

* The simplest expression a query can have is to just reference a column name. Here is its implementation:

```c++
class Column : public LogicalExpr {
public:
    std::string name;

    explicit Column(std::string n) : name(std::move(n)) {}

    Field toField(const LogicalPlan& input) const override {
        const auto& fields = input.schema().fields;
        auto it = std::find_if(fields.begin(), fields.end(),
            [this](const Field& f) { return f.name == this->name; }
        );

        if (it != fields.end()) {
            return *it;
        }

        std::stringstream err_msg;
        err_msg << "No column named '" << name << "' in [";
        for (size_t i = 0; i < fields.size(); ++i) {
            err_msg << fields[i].name;
            if (i < fields.size() - 1) err_msg << ", ";
        }
        err_msg << "]";

        throw std::invalid_argument(err_msg.str());
    }

    std::string toString() const override {
        return "#" + name;
    }
};
```

* `toField()` here is interesting. So what we are doing is basically trying to find if the referenced column exists in the input `const LogicalPlan& input`. For that, we first extract the fields of the input..

```c++
const auto& fields = input.schema().fields;
```

..And we then check if it exists using `std::find_if`, which is a powerful algorithm provided by the C++ STL (Standard Template Library). It takes a range of iterators—from `fields.begin()` to `fields.end()` —and searches for a match.

* What is particularly elegant here is our use of a C++11 lambda function:

```c++
auto it = std::find_if(fields.begin(), fields.end(),
    [this](const Field& f) { return f.name == this->name; }
);
```

* By passing `[this]` into the lambda's capture clause, we allow the inline function to access our Column class instance pointer. This lets us cleanly compare the input field's name (`f.name`) against our column's name (`this->name`) on the fly. If it is not present in the schema, the iterator returns `fields.end()`, and we construct a helpful error message.

* `toString()` is also the same, one thing is that the reason why we have a "#" is because we want to maintain a convention to distinguish column references from literal strings when printing plans.

### Column Index Expressions

* While referencing a column by its string name (like we did in the Column class) is great for human-readable SQL and initial logical planning, it isn't the most efficient way to access data during actual execution. Scanning arrays and doing string comparisons is slow.

* This is where `ColumnIndex` comes in:

```c++
class ColumnIndex : public LogicalExpr {
public:
    int i;

    explicit ColumnIndex(int index) : i(index) {}

    Field toField(const LogicalPlan& input) const override {
        return input.schema().fields[i];
    }

    std::string toString() const override {
        return "#" + std::to_string(i);
    }
};
```

* Compared to the `Column` class, `ColumnIndex` is delightfully simple. Instead of scanning through the input schema's fields using `std::find_if`, it resolves the field via direct array access (`input.schema().fields[i]`). This `O(1)` lookup is exactly what we want deep inside our execution loops. A mature query engine will typically run an "analyzer" or "binder" pass that converts all named Column references into these highly efficient ColumnIndex references right before physical execution begins.

* Our `toString()` method maintains the same `#` convention to denote a column, but instead of printing something like `#user_id`, it will print `#0` or `#3`.

### Literal Expressions

* Expressions like `salary * 1.1` in our previous example need to represent the literal value `1.1`. Which is why we need Literal Expressions for every data type. This is what we will be defining.

```c++
class LiteralString : public LogicalExpr {
public:
    std::string str;
    explicit LiteralString(std::string val) : str(std::move(val)) {}

    Field toField(const LogicalPlan& input) const override {
        return Field(str, ArrowTypes::StringType);
    }
    std::string toString() const override {
        return "'" + str + "'";
    }
};

class LiteralLong : public LogicalExpr {
public:
    int64_t n;
    explicit LiteralLong(int64_t val) : n(val) {}

    Field toField(const LogicalPlan& input) const override {
        return Field(std::to_string(n), ArrowTypes::Int64Type);
    }
    std::string toString() const override {
        return std::to_string(n);
    }
};

class LiteralFloat : public LogicalExpr {
public:
    float n;
    explicit LiteralFloat(float val) : n(val) {}

    Field toField(const LogicalPlan& input) const override {
        return Field(std::to_string(n), ArrowTypes::FloatType);
    }
    std::string toString() const override {
        return std::to_string(n);
    }
};
```

* We have `LiteralString`, `LiteralLong` and `LiteralFloat` which represent strings, 64-bit integers, and float types respectively.

* All of the `toField()` functions return a `Field` object to define the schema. Notice what we are passing as the first argument (the field name) to the `Field` constructor: we are passing the literal value itself!

* This is a standard query engine quirk. If you write a query like `SELECT 5`, the output schema needs a name for that column, so it creates a column literally named "5" mapped to an `ArrowTypes::Int64Type`. We also ensure that `LiteralString::toString()` wraps strings in single quotes (`'`) to maintain standard SQL printing conventions.

### Unary and Binary Expressions

* Next, we have our operator expressions: `UnaryExpr` and `BinaryExpr`. These represent mathematical or boolean operations (like `NOT A`, or `A + B`).

* Just like how a `LogicalPlan` forms a tree of table operations, these expressions form a tree of column operations. A `UnaryExpr` takes a single child expression (e.g., the `A` in `NOT A`), while a `BinaryExpr` takes a left (`l`) and a right (`r`) child expression.

* Here is their implementation:

```c++
class UnaryExpr : public LogicalExpr {
public:
    std::string name;
    std::string op;
    std::shared_ptr<LogicalExpr> expr;

    UnaryExpr(std::string n, std::string o, std::shared_ptr<LogicalExpr> e)
        : name(std::move(n)), op(std::move(o)), expr(std::move(e)) {}

    std::string toString() const override {
        return op + " " + expr->toString();
    }
};

class BinaryExpr : public LogicalExpr {
public:
    std::string name;
    std::string op;
    std::shared_ptr<LogicalExpr> l;
    std::shared_ptr<LogicalExpr> r;

    BinaryExpr(std::string n, std::string o, std::shared_ptr<LogicalExpr> left, std::shared_ptr<LogicalExpr> right)
        : name(std::move(n)), op(std::move(o)), l(std::move(left)), r(std::move(right)) {}

    std::string toString() const override {
        return l->toString() + " " + op + " " + r->toString();
    }
};
```

* If you look closely, you might notice something missing: neither of these classes implements the `toField()` method required by `LogicalExpr`!

* This is intentional. In C++, because we didn't override the pure virtual `toField()` method, these classes remain abstract. A generic `BinaryExpr` doesn't know what its return type should be. For example, `A = B` should return a Boolean type, while `A + B` should return a Numeric type. By leaving `toField()` abstract, we force specific operator subclasses (which we will build later, like `Eq`, `Neq`, `Add`, `And`) to define their own return types.

### Concrete Logical Plan Operators

* Now that we have our expression tree defined to handle row-level computations, we can build the concrete `LogicalPlan` nodes that actually transform datasets. Every SQL clause maps to one of these relational operators, forming a tree from the bottom (reading raw data) up to the final result.

#### 1. Scan

* Every query tree has to start somewhere, and `Scan` is always a leaf node. It wraps a `DataSource` (like our `CsvDataSource`) and takes an optional `projection` list of column names.

* Because it is at the very bottom of the tree, `children()` returns an empty vector. Its `schema()` simply delegates to the underlying `DataSource` (or narrows it down if a projection pushdown list was provided).

```c++
class Scan : public LogicalPlan {
public:
    std::string path;
    std::shared_ptr<DataSource> dataSource;
    std::vector<std::string> projection;

    Scan(std::string p, std::shared_ptr<DataSource> ds, std::vector<std::string> proj)
        : path(std::move(p)), dataSource(std::move(ds)), projection(std::move(proj)) {}

    Schema schema() const override {
        if (projection.empty()) {
            return dataSource->schema();
        }
        return dataSource->schema().select(projection);
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {}; // Leaf node has no inputs
    }

    std::string toString() const override {
        if (projection.empty()) {
            return "Scan: " + path + "; projection=None";
        }
        // Formats the projected columns...
        return "Scan: " + path + "; projection=[...]";
    }
};
```

#### 2. Projection

* `Projection` corresponds to the `SELECT` clause. It takes a single child `LogicalPlan` and a list of `LogicalExpr` objects (such as `#name` or `#salary * 1.1`).

* Unlike `Scan` or `Selection`, a `Projection` fundamentally changes the shape of the data. To compute its output `schema()`, it iterates through its list of expressions and calls `expr->toField(*input)` on each one, building a brand-new `Schema` out of the resulting fields.

```c++
class Projection : public LogicalPlan {
public:
    std::shared_ptr<LogicalPlan> input;
    std::vector<std::shared_ptr<LogicalExpr>> expr;

    Projection(std::shared_ptr<LogicalPlan> in, std::vector<std::shared_ptr<LogicalExpr>> e)
        : input(std::move(in)), expr(std::move(e)) {}

    Schema schema() const override {
        std::vector<Field> fields;
        fields.reserve(expr.size());
        for (const auto& e : expr) {
            fields.push_back(e->toField(*input));
        }
        return Schema(fields);
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {input};
    }

    std::string toString() const override {
        return "Projection: ...";
    }
};
```

#### 3. Selection (Filter)

* `Selection` corresponds to the `WHERE` clause (often called `Filter` in query plans). It wraps an `input` plan and a single boolean `LogicalExpr` predicate (e.g., `#department = 'Engineering'`).

* Because filtering only removes rows without altering columns, `Selection::schema()` simply passes through `input->schema()` untouched.

```c++
class Selection : public LogicalPlan {
public:
    std::shared_ptr<LogicalPlan> input;
    std::shared_ptr<LogicalExpr> expr;

    Selection(std::shared_ptr<LogicalPlan> in, std::shared_ptr<LogicalExpr> e)
        : input(std::move(in)), expr(std::move(e)) {}

    Schema schema() const override {
        return input->schema();
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {input};
    }

    std::string toString() const override {
        return "Selection: " + expr->toString();
    }
};
```

#### 4. Aggregate

* `Aggregate` handles `GROUP BY` clauses alongside aggregate functions like `SUM`, `MIN`, `MAX`, and `AVG`. It takes an `input` plan, a vector of grouping expressions (`groupExpr`), and a vector of aggregate expressions (`aggregateExpr`).

* Its output `schema()` is constructed by combining the fields produced by the grouping columns first, followed by the fields produced by the aggregate expressions.

```c++
class Aggregate : public LogicalPlan {
public:
    std::shared_ptr<LogicalPlan> input;
    std::vector<std::shared_ptr<LogicalExpr>> groupExpr;
    std::vector<std::shared_ptr<AggregateExpr>> aggregateExpr;

    Aggregate(std::shared_ptr<LogicalPlan> in,
              std::vector<std::shared_ptr<LogicalExpr>> groups,
              std::vector<std::shared_ptr<AggregateExpr>> aggs)
        : input(std::move(in)), groupExpr(std::move(groups)), aggregateExpr(std::move(aggs)) {}

    Schema schema() const override {
        std::vector<Field> fields;
        for (const auto& e : groupExpr) {
            fields.push_back(e->toField(*input));
        }
        for (const auto& e : aggregateExpr) {
            fields.push_back(e->toField(*input));
        }
        return Schema(fields);
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {input};
    }

    std::string toString() const override {
        return "Aggregate: ...";
    }
};
```

#### 5. Join

* Unlike the unary operators above (`Projection`, `Selection`, `Aggregate`) that take a single `input`, `Join` is a binary operator that combines *two* child plans (`left` and `right`). It also specifies a `JoinType` (`Inner`, `Left`, `Right`) and a list of join key pairs (`on`).

* Notice how `children()` returns both `{left, right}` so tree walkers can traverse both branches. In `schema()`, we merge the fields from the left table with the fields from the right table (excluding duplicate right-side join keys in an `Inner` join).

```c++
enum class JoinType { Inner, Left, Right };

class Join : public LogicalPlan {
public:
    std::shared_ptr<LogicalPlan> left;
    std::shared_ptr<LogicalPlan> right;
    JoinType joinType;
    std::vector<std::pair<std::string, std::string>> on;

    Join(std::shared_ptr<LogicalPlan> l,
         std::shared_ptr<LogicalPlan> r,
         JoinType type,
         std::vector<std::pair<std::string, std::string>> keys)
        : left(std::move(l)), right(std::move(r)), joinType(type), on(std::move(keys)) {}

    Schema schema() const override {
        std::set<std::string> duplicateKeys;
        if (joinType == JoinType::Inner) {
            for (const auto& pair : on) {
                duplicateKeys.insert(pair.second);
            }
        }

        std::vector<Field> fields = left->schema().fields;
        for (const auto& f : right->schema().fields) {
            if (duplicateKeys.find(f.name) == duplicateKeys.end()) {
                fields.push_back(f);
            }
        }
        return Schema(fields);
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {left, right};
    }

    std::string toString() const override {
        return "Join: ...";
    }
};
```

#### 6. Limit

* Finally, `Limit` maps directly to the SQL `LIMIT` clause. It wraps a single `input` plan and stores an integer `limit` representing the maximum number of rows to return. Just like `Selection`, it does not modify the columns at all, so `schema()` simply forwards `input->schema()`.

```c++
class Limit : public LogicalPlan {
public:
    std::shared_ptr<LogicalPlan> input;
    int limit;

    Limit(std::shared_ptr<LogicalPlan> in, int lim)
        : input(std::move(in)), limit(lim) {}

    Schema schema() const override {
        return input->schema();
    }

    std::vector<std::shared_ptr<LogicalPlan>> children() const override {
        return {input};
    }

    std::string toString() const override {
        return "Limit: " + std::to_string(limit);
    }
};
```




## Putting It Together

* Now that we have all the building blocks—our `DataSource`, `LogicalExpr` nodes, and `LogicalPlan` operators—let's see how they compose together in C++ to represent a complete SQL query. This is a sort of rough compilation path to be honest, but this is how it will be broken down as an when a query is given through the Query Engine. 

* Going back to our original example:

```sql
SELECT name, salary * 1.1 AS new_salary
FROM employees
WHERE department = 'Engineering'
```

* Because a `LogicalPlan` is a tree where each parent operator wraps its child input, we construct the plan **bottom-up**, starting from the raw data scan and wrapping each subsequent relational operation around it:

```c++
// 1. Start at the leaf: scan the employees table (no projection pushdown yet)
auto scan = std::make_shared<Scan>("employees", employeeDataSource, std::vector<std::string>{});

// 2. Apply the WHERE filter on top of the scan
auto filter = std::make_shared<Selection>(
    scan,
    std::make_shared<Eq>(
        std::make_shared<Column>("department"),
        std::make_shared<LiteralString>("Engineering")
    )
);

// 3. Apply the SELECT projection on top of the filtered rows
auto project = std::make_shared<Projection>(
    filter,
    std::vector<std::shared_ptr<LogicalExpr>>{
        std::make_shared<Column>("name"),
        std::make_shared<Alias>(
            std::make_shared<Multiply>(
                std::make_shared<Column>("salary"),
                std::make_shared<LiteralDouble>(1.1)
            ),
            "new_salary"
        )
    }
);
```

* Calling `project->pretty()` walks this tree recursively via `children()` and outputs the exact logical plan hierarchy:

```text
Projection: #name, #salary * 1.1 AS new_salary
  Selection: #department = 'Engineering'
    Scan: employees; projection=None
```

* Look at how much power this tree structure gives us before we even touch a single row of data:

  1. **Validation**: Calling `project->schema()` cascades down the tree, calling `toField()` on every expression to verify that `department`, `name`, and `salary` actually exist in `employees` and that `salary * 1.1` is a valid numeric operation.
  
  2. **Optimization**: An optimizer pass can inspect this tree and see that `Scan` is reading all columns (`projection=None`), even though `Projection` and `Selection` only ever use `name`, `salary`, and `department`. It can rewrite the `Scan` node to push that projection directly down to the CSV reader!
  
  3. **Execution**: Once validated and optimized, this logical tree is ready to be translated into a physical execution plan.
  
  
  

## Epilogue


* That is it for now. With `DataSources`, `LogicalExpr`, and our core `LogicalPlan` operators in place, Lattice now has a complete front-to-mid-end representation of a SQL query. We can take raw files on disk, infer their schemas, and build a validated, strongly-typed tree describing *what* computation needs to happen without worrying about *how* the CPU will execute it.

* Logical Plans was a fun module to build into Lattice. As someone with compiler experience it was interesting to see how it parallels a production compiler in terms of AST parsing and so on, writing out various combinations of possible expressions which all accumulate into one giant, optimized plan. 

* In the next part of this series, we will look at **Query Optimization** (rewriting these logical trees for performance, like pushing projections down into scans) and **Physical Plans**, where we finally start executing these trees over Apache Arrow `RecordBatch` streams. 




## References 

* None as far as I know of, I only read this cool blog on a [compile time CSV parser](https://blog.shaduri.dev/writing-a-compile-time-csv-parser-in-c-part-2-the-parser) which was cool. 
